#!/usr/bin/env python3

import os
import math
import random

import numpy as np

from utils.pareto import pareto

class ASHA:
    def __init__(self, hpm_dir, p, max_rung, logging, multi_objective=True):
        self.p = p
        self.multi_objective = multi_objective
        hpm_paths = [os.path.join(hpm_dir, hpm_name) for hpm_name in os.listdir(hpm_dir)]
        self.n = len(hpm_paths)
        self.i2h = {i: h for (i,h) in enumerate(hpm_paths)}
        self.i2n = {i: int(os.path.basename(h)[:-4]) for (i,h) in enumerate(hpm_paths)}
        self.configs = list(range(len(hpm_paths)))
        self.blacklist = []
        self.current_run = []
        self.max_rung = max_rung
        self.logging = logging
        self.rung_states = {}
        for rung in range(self.max_rung+1):
            self.rung_states[rung] = {}
            self.rung_states[rung]['finished'] = []
            self.rung_states[rung]['running'] = []
        self.config_states = {}
        for config in self.configs:
            self.config_states[config] = {}
            self.config_states[config]['rung'] = -1
            self.config_states[config]['bleu'] = -1
            self.config_states[config]['gpu_time'] = -1
            self.config_states[config]['bleus'] = []
            self.config_states[config]['gpu_times'] = []
            self.config_states[config]['converged'] = False
    
    def __getstate__(self):
        return dict(filter(lambda x: x[0]!='logging', self.__dict__.items()))

    def __setstate__(self, d):
        self.__dict__ = d

    def _add_to_list(self, item, lst):
        if item not in lst:
            lst.append(item)
        return lst
    
    def _remove_from_list(self, item, lst):
        if item in lst:
            lst.remove(item)
        return lst

    def _get_rung_blacklist(self, r):
        blk_r = []
        for b in self.blacklist:
            if self.config_states[str(b)]['rung'] == r:
                blk_r.append(b)
        return blk_r
    
    def tops(self, rung, finished_configs):
        '''
        rung: the current rung
        finished_configs: finished configs in the current rung

        Return the top s config candidates to that can be moved to the next rung.
        '''
        topk_to_next_rung = math.floor(self.n*self.p**(-rung-1))
        blacklist_this_rung = self._get_rung_blacklist(rung)
        n_this_rung = math.floor(self.n*self.p**(-rung)) - len(blacklist_this_rung)
        if len(finished_configs) > n_this_rung - topk_to_next_rung:
            tops_to_next_rung = min(len(finished_configs) - (n_this_rung-topk_to_next_rung), \
                topk_to_next_rung)
            # print("****topk_to_next_rung ", topk_to_next_rung)
            # print("****n_this_rung ", n_this_rung)
            # print("****tops_to_next_rung ", topk_to_next_rung)
            bleu_lst = [] 
            gpu_time_lst = []
            finished_configs = set(finished_configs) - set(self.blacklist)
            for config in list(finished_configs):
                bleu_lst.append(self.config_states[config]['bleus'][rung])
                gpu_time_lst.append(self.config_states[config]['gpu_times'][rung])
            if self.multi_objective:
                y = np.stack((bleu_lst, gpu_time_lst))
                ranks = np.array(pareto(y, opt=[1, -1]))
                tops_cids = np.argpartition(ranks,\
                    -tops_to_next_rung)[-tops_to_next_rung:]
                # print("*****rung", rung, "tops y: ", y)
                # print("*****rung", rung, "tops ranks: ", ranks)
                # print("*****rung", rung, "tops_cids: ", tops_cids)
            else:
                tops_cids = np.argpartition(np.array(bleu_lst),\
                    -tops_to_next_rung)[-tops_to_next_rung:]
            tops_configs = [list(finished_configs)[c] for c in range(len(finished_configs)) if c in tops_cids]
            return tops_configs
        return []
                
    def get_candidates(self):
        candidates = set()
        print("Obtaining the candidates .....")
        print("Rung states: ", self.rung_states)
        print("Config states: ", self.config_states)
        for rung in range(self.max_rung+1):
            finished = self.rung_states[rung]['finished']
            running = self.rung_states[rung]['running']
            if rung < self.max_rung:
                next_finished = self.rung_states[rung+1]['finished']
                next_running = self.rung_states[rung+1]['running']
            else:
                next_finished = []
                next_running = []
            if rung > 0:
                pre_finished = set(self.rung_states[rung-1]['finished'])
                tops_finished = self.tops(rung-1, pre_finished)
            else:
                tops_finished = set(self.configs) - set(running) - set(finished) - set(self.blacklist)
            candidates_this_rung = set(tops_finished) - set(next_finished) - set(next_running) \
                 - set(finished) - set(running)
            print("rung ", rung, "candidates ", candidates_this_rung)
            candidates = candidates.union(candidates_this_rung)
        candidates = list(candidates)
        return candidates
    
    def pick_next_candidate(self, candidates):
        return random.choice(candidates)

    def promote(self, config):
        prev_rung = self.config_states[config]['rung']
        if (prev_rung != -1 and config in self.rung_states[prev_rung]['finished']) \
            or prev_rung == -1:
            self.config_states[config]['rung'] += 1
        cur_rung = self.config_states[config]['rung']
        self.rung_states[cur_rung]['running'] = \
            self._add_to_list(config, self.rung_states[cur_rung]['running'])
        self.current_run = \
            self._add_to_list(config, self.current_run)
    
    def blacklist_config(self, config):
        self.blacklist = self._add_to_list(config, self.blacklist)
        cur_rung = self.config_states[config]['rung']
        self.rung_states[cur_rung]['running'] = \
            self._remove_from_list(config, self.rung_states[cur_rung]['running'])
        self.current_run = self._remove_from_list(config, self.current_run)

    def finish_config(self, config, bleu, gpu_time):
        self.config_states[config]['bleu'] = bleu
        self.config_states[config]['gpu_time'] = gpu_time
        self.config_states[config]['bleus'].append(bleu)
        self.config_states[config]['gpu_times'].append(gpu_time)
        cur_rung = self.config_states[config]['rung']
        self.rung_states[cur_rung]['running'] = \
            self._remove_from_list(config, self.rung_states[cur_rung]['running'])
        self.rung_states[cur_rung]['finished'] = \
            self._add_to_list(config, self.rung_states[cur_rung]['finished'])
        self.current_run = self._remove_from_list(config, self.current_run)
    
    def move_coverged_config_to_next_rung(self, config):
        cur_rung = self.config_states[config]['rung']
        if cur_rung < self.max_rung:
            self.config_states[config]['rung'] = cur_rung + 1
            self.rung_states[cur_rung+1]['finished'] = \
                self._add_to_list(config, self.rung_states[cur_rung+1]['finished'])
            self.config_states[config]['bleus'].append(self.config_states[config]['bleu'])
            self.config_states[config]['gpu_times'].append(self.config_states[config]['gpu_time'])
    
    def log_state(self):
        log_string = ""
        for rung in range(self.max_rung+1):
            finished = sorted(self.rung_states[rung]['finished'])
            if finished != []:
                log_string += '-'*20 + '\n'
                log_string += "Rung " + str(rung) + ":\n"
                num_line_groups = math.ceil(len(finished) / 10)
                for nlg in range(num_line_groups):
                    fin_str = ''
                    bleu_str = ''
                    gpu_str = ''
                    fin_id_str = ''
                    for fin in finished[nlg*10: (nlg+1)*10]:
                        fin_id_str += str(fin).ljust(10)
                        fin_str += str(self.i2n[fin]).ljust(10)
                        bleu_str += str(self.config_states[fin]['bleus'][rung]).ljust(10)
                        if self.multi_objective:
                            gpu_str += str(self.config_states[fin]['gpu_times'][rung]).ljust(10)
                    log_string += 'Finished Jobs'.ljust(16) + fin_str + "\n"
                    log_string += 'Ids'.ljust(16) + fin_id_str + "\n"
                    log_string += 'BLEU'.ljust(16) + bleu_str + "\n"
                    if gpu_str != '':
                        log_string += 'decoding time'.ljust(16) + gpu_str + "\n"
                    log_string += "\n"
        self.logging.info(str(log_string)) 

    def finish_asha(self):
        self.logging.info("ASHA finished successfully!")
        self.log_state()
        best_config = self.i2n[self.rung_states[self.max_rung]["finished"][0]]
        log_str = ""
        if self.blacklist != []:
            log_str += "Configs that failed training because of GPU Memory error:\n"
            log_str += ", ".join([str(self.i2n[b]) for b in self.blacklist])
            log_str += "\n"
        log_str += "Best config: " + str(best_config) + "\t" + \
            "BLEU: " + str(self.config_states[best_config]['bleu'])
        if self.multi_objective:
            log_str += "\t decoding time: " + \
                str(self.config_states[best_config]["gpu_time"])
        self.logging.info(str(log_str))
