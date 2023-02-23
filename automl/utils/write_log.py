import json

def init_log(args, logging):
    logging.info(str('Run ASHA with Arguments: ' + '\n' +
                 'minimum number of checkpoints (r): ' + str(args.r) + '\n' + 
                 'number of checkpoints per rung (u): ' + str(args.u) + '\n' +
                 'maximum checkpoints (R): ' + str(args.R) + '\n' +
                 'reduction rate (p): ' + str(args.p) + '\n' +
                 'number of GPUs (G): ' + str(args.G)))
    logging.info(str('work directory: ' + args.workdir))
    logging.info(str('job log directory: ' + args.job_log_dir))
    if args.multi_objective:
        logging.info('Multi-objective optimization: BLEU and decoding time will be optimized at the same time.')
    else:
        logging.info('Single-objective optimization: BLEU will be optimized.')


def save_asha_state(asha, jobmanager, save_path, logging):
    state_dict = {}
    state_dict['asha'] = asha.__getstate__()
    state_dict['jobmanager'] = jobmanager.__getstate__()
    with open(save_path, 'w') as f:
        json.dump(state_dict, f)
    logging.info(str("Saved ASHA states to " + save_path))
    
def load_asha_state(load_path, logging):
    with open(load_path) as f:
        state_dict = json.load(f)
    asha = state_dict['asha']
    keys = ['config_states', 'rung_states', 'i2h', 'i2n']
    for key in keys:
        asha[key] = dict(map(lambda x: (int(x[0]), x[1]), asha[key].items()))
    jm = state_dict['jobmanager']
    asha['logging'] = logging
    jm['logging'] = logging
    logging.info(str('Loaded ASHA states from: ' + load_path + "\n" + \
        'minimum number of checkpoints (r): ' + str(jm['r']) + '\n' +\
        'number of checkpoints per rung (u): ' + str(jm['u']) + '\n' +
        'maximum checkpoints (R): ' + str(jm['R']) + '\n' +
        'reduction rate (p): ' + str(asha['p']) + '\n'))
    return asha, jm
    
