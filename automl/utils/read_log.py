#!/usr/bin/env python3

import os
import glob
import math

from constants import *
from utils.hpm_file import get_valid_tok

def readfile(f):
    '''
    :param f: file path
    :return: list
    '''
    with open(f) as fobj:
        return fobj.readlines()

def check_train_states(model_path):
    '''
    Check the training states of a model.
    States: SUCCESS, GPU_ERROR, MEM_ERROR, RUNNING
    '''
    log_path = os.path.join(model_path, "log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
            lines.reverse()
        for l in lines:
            if "Maximum number of not improved checkpoints" in l:
                return CONVERGED
            elif "CUDA error: all CUDA-capable devices are busy or unavailable" in l:
                return GPU_ERROR
            elif "CUDA out of memory" in l:
                return MEM_ERROR
        if "Training finished" in lines[0]:
            return SUCCESS
        return RUNNING
    else:
        return NOTEXIST

def check_valid_states(model_path):
    '''
    Check whether the evaluation finished successfully.
    States: ERROR, RUNNING, SUCCESS
    '''
    log_path_lst = glob.glob(os.path.join(model_path, "*.1best.log"))
    if len(log_path_lst) == 1:
        log_path = log_path_lst[0]
        with open(log_path) as f:
            for l in f.readlines():
                if "Uncaught exception" in l:
                    return ERROR
                if "Processed" in l:
                    return SUCCESS
            return RUNNING
    else:
        return NOTEXIST

def extract_eval_metrics(model_path):
    '''
    Extract model evaluation results from decode files.
    The decoding results should be written under the corresponding model directory.
    :param model_path: The path to the model directory.
    :return: A dictionary of evaluation results.
    '''
    eval_metrics = {}
    vb_log_path = glob.glob(os.path.join(model_path, "*.1best.log"))[0]
    val_tok_path = get_valid_tok(os.path.join(model_path, 'hyperparams.txt'))
    bleu_line = os.popen("sacrebleu --tokenize none " + \
        val_tok_path + \
        " < " + vb_log_path[:-4]).read()
    bleu_path = os.path.join(model_path, os.path.basename(vb_log_path[:-4])+".bleu")
    with open(bleu_path, 'w') as f:
        f.write(bleu_line)
    #vb_path = glob.glob(os.path.join(model_path, "*.1best.bleu"))[0]
    # dev BLEU
    #eval_metrics["dev_bleu"] = float(readfile(vb_path)[0].strip().split()[2][:-1])
    eval_metrics["bleu"] = float(bleu_line.split('=')[1].split()[0])
    # dev GPU time
    eval_metrics["gpu_time"] = math.ceil(float(readfile(vb_log_path)[-3].strip().split(',')[0].split()[-1]))
    os.remove(vb_log_path)
    os.remove(bleu_path)
    return eval_metrics
