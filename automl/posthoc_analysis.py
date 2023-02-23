#!/usr/bin/env python3

# This script performs post-hoc analysis of model runs using Explainable Boosting Machines
# See: https://aclanthology.org/2022.blackboxnlp-1.5.pdf

import sys
import os 
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
from jenkspy import JenksNaturalBreaks



tracked_metrics = ['avg-sec-per-sent-val', 'bleu-val', 'perplexity-train', 'perplexity-val', 'time-elapsed']

def define_tracked_hparams(hparams_str):
    if hparams_str == None:
        # default
        tracked_hparams = ['bpe_symbols_src', 'bpe_symbols_trg', 'num_layers', 'transformer_model_size', 'transformer_attention_heads', 'transformer_feed_forward_num_hidden', 'initial_learning_rate', 'seed']
    else:
        tracked_hparams = hparams_str.split(':')

    feature_types=['continuous']*len(tracked_hparams)
    for i, f in enumerate(tracked_hparams):
        if f in set(['num_layers', 'seed']):
            feature_types[i] = 'nominal'

    return tracked_hparams, feature_types



def collect_logs(basedir, target_label_metric, tracked_hparams):
    stack = []
    label = []
    total_hours = 0
    for (dirname, subdirlist, filelist) in os.walk(basedir):
        filelist_set = set(filelist)
        if 'metrics' in filelist_set and 'hyperparams.txt' in filelist_set:
            metrics = read_metrics(os.path.join(dirname, 'metrics'))
            best_checkpoint = metrics.iloc[metrics[target_label_metric].idxmax()]
            label.append(best_checkpoint[target_label_metric])
            hparams = read_hparams(os.path.join(dirname, 'hyperparams.txt'), tracked_hparams)
            stack.append(hparams)
            total_hours += float(metrics.iloc[-1]['time-elapsed'])/3600
            #print(dirname, hparams)
        
    stack_hparams = pd.DataFrame(data=stack, dtype=str, columns=tracked_hparams)
    print(f"Total hours: {total_hours} \nTotal configurations: {len(stack)}")
    return (label, stack_hparams)


def read_metrics(filename):
    metrics = [] 
    with open(filename) as F:
        for line in F:
            fields = line.split()
            kv = dict(item.split('=') for item in fields[1:])
            metrics.append([kv[m] for m in tracked_metrics])
    return pd.DataFrame(data=metrics, dtype=float, columns=tracked_metrics)


def read_hparams(filename, tracked_hparams):
    hparams = {}
    with open(filename) as F:
        for line in F:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=')
                hparams[k] = v
    d = [hparams[k] for k in tracked_hparams]
    return d



def main(models_rootpath, output_path, ebm_pairwise, target_label_metric, jenks, split_seed, hparams_str=None):

    # 1. Define hyperparameters to track in EBM analysis
    tracked_hparams, feature_types = define_tracked_hparams(hparams_str)

    print("==== EBM post-hoc analysis ====")
    print(f"reading logs from {models_rootpath}")
    print(f"hparams: {tracked_hparams}")
    print(f"target: {target_label_metric} / discretize_with_jenks?: {jenks}")

    # 2. Read all logs
    print("\n# Reading model directories...")
    y, X = collect_logs(models_rootpath, target_label_metric, tracked_hparams)

    print("\n# Overall counts for each hparam:")
    for col in tracked_hparams: 
        print(X[col].value_counts(), end='\n\n')

    # 3. Apply target discretization, if desired
    if jenks != 0:
        jnb = JenksNaturalBreaks(jenks)
        jnb.fit(y)
        y=jnb.labels_

    print("\n# Percentile of target:")
    yy = np.array(y)
    for p in np.linspace(0, 100, 11):
        print("%d: %.4f" %(p, np.percentile(yy, p)))

    # 4. split train/test and fit EBM
    ebm_test_size=0.15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ebm_test_size, random_state=split_seed)

    ebm_seed = 1
    ebm = ExplainableBoostingRegressor(random_state=ebm_seed, interactions=ebm_pairwise, feature_types=feature_types)
    ebm.fit(X_train, y_train)

    print("\n# EBM test score (R^2): %.4f" % ebm.score(X_test, y_test))


    # 5. save visualizations
    print(f"# Saving EBM visualizations to directory: {output_path}")
    ebm_global = ebm.explain_global()
    plotly_fig = ebm_global.visualize()
    plotly_fig.write_image(f"{output_path}/ebm-global.png")
    for index, value in enumerate(ebm.term_features_):
        plotly_fig = ebm_global.visualize(index)
        plotly_fig.write_image(f"{output_path}/ebm-feature-{index}.png")

    local_test_n = min(4, len(y_test))
    ebm_local = ebm.explain_local(X_test[:local_test_n], y_test[:local_test_n])
    for index in range(local_test_n):
        plotly_fig = ebm_local.visualize(index)
        plotly_fig.write_image(f"{output_path}/ebm-local-{index}.png") 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run post-hoc analysis with Explainable Boosting Machines (EBM)')
    parser.add_argument('-m', type=str, required=True, help='Root path of all model directories to be analyzed')    
    parser.add_argument('-d', type=str, required=True, help='Directory to save EBM visualizations')
    parser.add_argument('-i', type=int, default=5, help='# of EBM pairwise interaction features')
    parser.add_argument('-t', type=str, default='bleu-val', help='Target label for EBM: {bleu-val, perplexity-train/val, avg-sec-per-sent-val}')
    parser.add_argument('-j', type=int, default=0, help='Discretize target to N intergers using Jenks. N=0 means no discretize, use raw values')
    parser.add_argument('-r', type=int, default=1, help='Random seed for EBM train/test split')
    parser.add_argument('-s', type=str, default=None, help='colon-separated list of hparam to analyze')

    args = parser.parse_args()
    main(args.m, args.d, args.i, args.t, args.j, args.r, args.s)

