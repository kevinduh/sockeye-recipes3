## Hypeparameter Optimization with Grid Search

This recipe demonstrates how to setup grid search, trying all combinations of hyperparameters.

First, we define the space of hyperparameters in `hpm-space.yaml`. This is a simple YAML file that lists the hyperparameters and their desired choices as lists.

```
workdir: ./models 
rootdir: ../../   
src: zh           
trg: en
train_tok: ../ted/multitarget-ted/en-zh/tok/ted_train_en-zh.tok.clean
valid_tok: ../ted/multitarget-ted/en-zh/tok/ted_dev_en-zh.tok
bpe_symbols_src:
- 30000
- 10000
bpe_symbols_trg:
- 30000
encoder: transformer
decoder: transformer
num_layers:
- "4:2"
- "6:3"
transformer_model_size: [512, 1024]
transformer_attention_heads: 8
...
```

The first 6 elements here (workdir, rootdir, src, trg, train_tok, valid_tok) must be defined as single values, but others can be lists. For example, here we see bpe_symbols_src can range as 30000 or 10000, while bpe_symbols_trg is fixed at 30000. Similarly, num_layers can be either "4:2" or "6:3" and transformer_model_size can be either 512 or 1024. 

Now, we can start preparation for grid search. Do the following:

```
mkdir -p models/data-bpe
conda activate sockeye3
python3 ../../automl/generate_hpm.py hpm-space.yaml
```

This will print out something like this:

```
Step1: run the following to prepare the data: /bin/sh prep.sh
Step2: 16 hpm files generated in ./models
Step3: run the following to train...
qsub -S /bin/bash -V -cwd -q gpu.q@@2080 -l gpu=1,h_rt=12:00:00,num_proc=1 -j y -t 1:16 -tc 5 sge-train.sh
```

Following the instructions, we first run `/bin/sh prep.sh`. This will run preprocess-bpe.sh and sockeye.prepare_data and save to models/data-bpe. It is more efficient to have this done all at once, before running the training part of grid search. 

Next, we can look at the generated hpm files in `models/*.hpm`. There are 16 in total, corresponding to the cross product of two BPE choices, two layer choices, two model size choices, and two learning rate choices specified in the example `hpm-space.yaml`. 

Finally, run the `qsub` command (or whichever variant that is suitable on your system) to run train these models. 

## Post-hoc analysis

Given a group of trained model directories, we may wish to understand better how the hyperparameters interact with each other and with BLEU score. 
We provide a simple script `posthoc_analysis.py` to do this using Explainable Boosting Machines (EBM).. 

First, we need to install some dependencies:
```
conda activate sockeye3
/bin/sh $sockeye_root/install/install_3rdparty.sh
```

Next, simply run:
```
mkdir ebm
python $sockeye_root/automl/posthoc_analysis.py -m models -d ebm
```

This will read all the training logs from models/ subdirectory and save the EBM visualizations to ebm/
See [Post-Hoc Interpretation of Transformer Hyperparameters with Explainable Boosting Machines](https://aclanthology.org/2022.blackboxnlp-1.5.pdf) for details.

## Cleanup

Grid search and various AutoML methods like ASHA may generate many subdirectories, taking up much disk space. 
Use `scripts/cleanup.py` to delete some of the files in models/ programmatically.
