#!/usr/bin/env python3

import os
import shutil
import subprocess
import time
import datetime

from constants import *
from utils.hpm_file import update_hpm
from utils.read_log import check_train_states, check_valid_states

class JobManager:
    def __init__(self, num_gpu, r, u, R, logging):
        self.num_gpu = num_gpu
        self.r = r
        self.u = u
        self.R = R
        self.logging = logging
    
    def __getstate__(self):
        return dict(filter(lambda x: x[0]!='logging', self.__dict__.items()))

    def __setstate__(self, d):
        self.__dict__ = d

    def initialize_hpms(self, hpm_dir, workdir):
        hpm_names = os.listdir(hpm_dir)
        hpm_ids = [hn[:-4] for hn in hpm_names]
        os.makedirs(os.path.join(workdir, 'models'), exist_ok=True)
        for i in range(len(hpm_names)):
            modeldir = os.path.join(workdir, 'models', hpm_ids[i])
            hpm_path = os.path.join(hpm_dir, hpm_names[i])
            update_dict = {'workdir': workdir,
                           'modeldir': modeldir,
                           'max_checkpoints': str(self.r)}
            update_hpm(hpm_path, update_dict)
    
    def qdel(self, hpm, modeldir):
        with subprocess.Popen(["qstat"], stdout=subprocess.PIPE) as proc:
            qstat_lines = proc.stdout.read().decode('utf-8')
        for line in qstat_lines.split('\n'):
            if "gpu.q" in line and "asha" in line and os.path.basename(hpm)[:-4] in line:
                if os.path.exists(os.path.join(modeldir, "log")):
                    shutil.move(os.path.join(modeldir, "log"), \
                        os.path.join(modeldir, "log_"+"T".join(str(datetime.datetime.now()).split())))
                job_id = line.split()[0]
                os.system("qdel " + job_id)
                time.sleep(15)

    def qsub_train(self, job_log_dir, hpm):
        with open("qsub.sh") as f:
            qsub_line = f.readlines()[1].strip()
        qsub_line = qsub_line.format(os.path.join(job_log_dir, os.path.basename(hpm)[:-4]+'_train'),
                        'ashat'+os.path.basename(hpm)[:-4],
                        hpm)
        self.logging.info(str(qsub_line))
        os.system(qsub_line)
        time.sleep(15)
    
    def qsub_val(self, job_log_dir, hpm, modeldir):
        # make sure there's a params.best in modeldir
        params = [p for p in os.listdir(modeldir) if p.startswith('params')]
        if "params.best" not in params:
            params.sort(key=lambda p: int(p[-5:]))
            os.system("ln -s " + modeldir + "/" + params[-1] + " " + modeldir + "/params.best")
        with open("qsub.sh") as f:
            qsub_line = f.readlines()[3].strip()
        qsub_line = qsub_line.format(os.path.join(job_log_dir, os.path.basename(hpm)[:-4]+'_val'),
                        'ashav'+os.path.basename(hpm)[:-4],
                        hpm, modeldir)
        self.logging.info(str(qsub_line))
        os.system(qsub_line)
        time.sleep(15)
    
    def update_hpm_to_next_rung(self, hpm, target_rung):
        pree = min(self.r + self.u * (target_rung-1), self.R)
        maxe = min(self.r + self.u * target_rung, self.R) - pree
        dict = {'max_checkpoints': maxe}
        update_hpm(hpm, dict)

    def num_avail_gpus(self):
        with subprocess.Popen(["qstat"], stdout=subprocess.PIPE) as proc:
            qstat_lines = proc.stdout.read().decode('utf-8')
        num_run_gpus = 0
        for line in qstat_lines.split('\n'):
            if "gpu.q" in line and "asha" in line:
                num_run_gpus += 1
        return self.num_gpu - num_run_gpus
    
    def check_gpu_states(self, jobname):
        with subprocess.Popen(["qstat"], stdout=subprocess.PIPE) as proc:
            qstat_lines = proc.stdout.read().decode('utf-8')
        for line in qstat_lines.split('\n'):
            line = line.split()
            if jobname in line:
                if "qw" in line:
                    return WAIT
                elif "r" in line:
                    return RUNNING
        return GPUNOTEXIST
    
    def submit_job(self, job_log_dir, hpm, modeldir):
        train_job_state = check_train_states(modeldir)
        val_job_state = check_valid_states(modeldir)
        train_gpu_state = self.check_gpu_states("ashat"+os.path.basename(modeldir))
        val_gpu_state = self.check_gpu_states("ashav"+os.path.basename(modeldir))
        print("config " + os.path.basename(modeldir), \
            "train_job_state: ", train_job_state, \
            "val_job_state: ", val_job_state,
            "train_gpu_state: ", train_gpu_state, \
            "val_gpu_state: ", val_gpu_state)
        if train_gpu_state == WAIT or val_gpu_state == WAIT:
            return 0
        elif (train_job_state == SUCCESS or train_job_state == CONVERGED) \
             and (val_job_state == NOTEXIST or val_job_state == ERROR):
            self.qdel(hpm, modeldir)
            if self.num_avail_gpus() > 0:
                self.qsub_val(job_log_dir, hpm, modeldir)
            return 0
        elif (train_job_state == SUCCESS or train_job_state == CONVERGED) \
            and val_job_state == RUNNING:
            return 0
        elif train_job_state == SUCCESS and val_job_state == SUCCESS:
            return 1
        elif train_job_state == CONVERGED and val_job_state == SUCCESS:
            return 2
        elif train_job_state == GPU_ERROR:
            self.qdel(hpm, modeldir)
            if self.num_avail_gpus() > 0:
                self.qsub_train(job_log_dir, hpm)
            return 0
        elif train_job_state == MEM_ERROR:
            self.qdel(hpm, modeldir)
            return -1
        elif train_job_state == RUNNING or train_job_state == NOTEXIST:
            return 0
        elif train_job_state == STORAGE_ERROR or val_job_state == STORAGE_ERROR:
            print("Storage error occurred for " + hpm + \
                ". The current run of automl will end shortly." \
                "Please make sure there's enough space on your disk before resume the job.")
            self.logging.info("The job is interrupted because there's not enough space on the disk.")
            self.qdel(hpm, modeldir)
            return 3
