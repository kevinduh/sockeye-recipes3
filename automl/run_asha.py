import argparse
import os
from threading import Thread, Event
import math

from constants import *
from utils.read_log import extract_eval_metrics
from job_manager import JobManager
from asha import ASHA
from utils.write_log import init_log, save_asha_state, load_asha_state
from utils.check_args import check_args

class AutomlThread(Thread):
    def __init__(self, event, func, args, jobmanager, asha):
        Thread.__init__(self)
        self.stopped = event
        self.func = func
        self.args = args
        self.jobmanager = jobmanager
        self.asha = asha

    def run(self):
        while not self.stopped.wait(self.args.timer_interval):
            if self.stopped.is_set():
                break
            finished = self.func(self.args, self.jobmanager, self.asha)
            if finished:
                self.stopped.set()

def run_asha(args, jobmanager, asha):
    finished_asha = False
    print("config id to real id: ", asha.i2n)
    current_run = list(asha.current_run)
    #print("Current run to check states: ", " ".join([str(x) for x in current_run]))
    for config in current_run:
        hpm = asha.i2h[config]
        modeldir = os.path.join(args.workdir, 'models', os.path.basename(hpm[:-4])) 
        state = jobmanager.submit_job(args.job_log_dir, hpm, modeldir)
        if state == -1:
            asha.blacklist_config(config)
        elif state > 0:
            metrics = extract_eval_metrics(modeldir)
            asha.finish_config(config, metrics['bleu'], metrics['gpu_time'])
            if state == 1:
                jobmanager.update_hpm_to_next_rung(hpm, asha.config_states[config]['rung']+1)
            elif state == 2:
                asha.config_states[config]['converged'] = True

    num_avail_gpu = jobmanager.num_avail_gpus()
    print("\nnum_avail_gpu ", num_avail_gpu)
    for _ in range(num_avail_gpu):
        config_converged = True
        while config_converged:
            candidates = asha.get_candidates()
            print("asha.current_run: ", asha.current_run)
            #print("Candidates: ", " ".join([str(c) for c in candidates]))
            if candidates == []:
                if asha.current_run == []:
                    asha.finish_asha()
                    finished_asha = True
                    return finished_asha
                next_cand = None
                config_converged = False
            else:
                next_cand = asha.pick_next_candidate(candidates)
                print("Picked candidate: ", next_cand)
                if asha.config_states[next_cand]['converged'] == True:
                    asha.move_converged_config_to_next_rung(next_cand)
                else:
                    config_converged = False
        if next_cand is not None:
            asha.promote(next_cand)
            hpm = asha.i2h[next_cand]
            jobmanager.qsub_train(args.job_log_dir, hpm)
        print("current run after promotion: ", asha.current_run)
        print("\n")
    asha.log_state()
    save_asha_state(asha, jobmanager, args.ckpt, asha.logging)
    return finished_asha

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization with \
                                    Asynchronous Successive Halving (ASHA).')
    parser.add_argument('-r', type=int, 
                        help='Minimum resource (number of checkpoints) allocated to each config. Unit: ')
    parser.add_argument('-u', type=int,
                        help='Resource (number of checkpoints) allocated to a config within a rung.')
    parser.add_argument('-R', type=int,
                        help='Maximum resource (number of checkpoints) allocated to a config.')
    parser.add_argument('-p', type=int,
                        help='Reduction rate. Pick top 1/p configs to move to next rung.')
    parser.add_argument('-G', type=int,
                        help='Number of available GPUs.')
    parser.add_argument('--timer-interval', type=int, default=60,
                        help="Check the job states every n seconds.")
    parser.add_argument('--workdir', type=str,
                        help='Directory to save related files generated by this run of automl.')
    parser.add_argument('--multi-objective', default=False, action='store_true',
                        help="Whether to optimize multiple objectivs (dev_bleu & dev_gpu_time);\
                            if set to False, will only evaluate dev_bleu.")
    parser.add_argument('--job-log-dir', help="Directory to save all the job logs.")
    parser.add_argument('--resume-from-ckpt', type=str, default=None, help="Resume from previous ASHA state.")
    parser.add_argument('--ckpt', type=str, required=True, help='Save ASHA states to the path.')
    args = parser.parse_args()
    check_args(args)

    import logging
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(handlers=[
                            logging.FileHandler(os.path.join(args.workdir, 'log')),
                            logging.StreamHandler()],                   
                        level=logging.INFO, format=FORMAT)
    init_log(args, logging)

    hpm_dir = os.path.join(args.workdir, 'hpms')

    jobmanager = JobManager(num_gpu=args.G, r=args.r, u=args.u, R=args.R, logging=logging)
    jobmanager.initialize_hpms(hpm_dir, args.workdir)

    n = len(os.listdir(hpm_dir))
    max_rung = min(math.floor(math.log(n, args.p)), 
                math.ceil((args.R-args.r)/args.u)+1)
    asha = ASHA(hpm_dir=hpm_dir, p=args.p, max_rung=max_rung, logging=logging, \
                multi_objective=args.multi_objective)

    if args.resume_from_ckpt != None and os.path.exists(args.resume_from_ckpt):
        asha_state_dict, jobmanager_state_dict = \
            load_asha_state(args.resume_from_ckpt, logging)
        asha.__setstate__(asha_state_dict)
        jobmanager.__setstate__(jobmanager_state_dict)
 
    stopFlag = Event()
    automl_thread = AutomlThread(stopFlag, run_asha, args, jobmanager, asha)
    automl_thread.start()

if __name__ == "__main__":
    main()