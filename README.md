# sockeye-recipes3

Training scripts, recipes, and hyperparameter optimization for the Sockeye Neural Machine Translation (NMT) toolkit
- The original Sockeye codebase is at [AWS Labs](https://github.com/awslabs/sockeye). This repo is based off [a stable fork](https://github.com/kevinduh/sockeye), version: 3.1.14
- Here we focus on Sockeye v3. This repo is similar but not exactly back-compatible with the older version of [sockeye-recipes for Sockeye v2](https://github.com/kevinduh/sockeye-recipes2).

This repo contains scripts that makes it easy to run and replicate NMT experiments.
All model hyperparameters are documented in a file "hyperparams.txt", which are passed to different steps in the pipeline:
- scripts/preprocess-bpe.sh: Preprocess bitext via subword segmentation
- scripts/train.sh: Train the NMT model given bitext
- scripts/translate.sh: Translates a tokenized input file using an existing model

We also implement Hyperparameter Optimization methods (AutoML) to make it easy to efficiently find a near-optimal model for a given dataset. 

## Installation
First, clone this package: 
```bash
git clone https://github.com/kevinduh/sockeye-recipes3.git sockeye-recipes3
```

We assume that Anaconda for Python virtual environments is available on the system.
All code needed by the scripts are installed in an Anaconda environment named `sockeye3`:

```bash
cd path/to/sockeye-recipes3
bash ./install/install_sockeye_default.sh
```

If you need to backup or remove your Anaconda environment before re-installing: 
```bash
conda create --name sockeye3_bkup --clone sockeye3
conda remove --name sockeye3 --all
```

## Recipes 

The [egs](egs) subdirectory contains recipes for various datasets. 

* [egs/quickstart](egs/quickstart): For first time users, this recipe explains how sockeye-recipe works. 

* [egs/ted](egs/ted): Recipes for training various NMT models, using a TED Talks dataset consisting of 20 different languages.

The [hpm](hpm) subdirectory contains hyperparameter (hpm) file templates. Besides NMT hyerparameters, the most important variables in this file to set are below: 

* rootdir: location of your sockeye-recipes installation, used for finding relevant scripts (i.e. this is current directory, where this README file is located.)

* modeldir: directory for storing a single Sockeye model training process

* workdir: directory for placing various modeldirs (i.e. a suite of experiments with different hyperparameters) corresponding to the same dataset

* train_tok and valid_tok: prefix of tokenized training and validation bitext file path

* train_bpe_{src,trg} and valid_bpe_{src,trg}: alternatively, prefix of the above training and validation files already processed by BPE


## Design Principles and Suggested Usage

Building NMT systems can be a tedious process involving lenghty experimentation with hyperparameters. The goal of sockeye-recipes is to make it easy to try many different configurations and to record best practices as example recipes. The suggested usage is as follows:
- Prepare your training and validation bitext beforehand with the necessary preprocessing (e.g. data consolidation, tokenization, lower/true-casing). Sockeye-recipes simply assumes pairs of train_tok and valid_tok files. 
- Set the working directory to correspond to a single suite of experiments on the same dataset (e.g. WMT17-German-English)
- Run preprocess-bpe.sh with different BPE vocabulary sizes (bpe_symbols_src, bpe_symbols_trg). These can be saved all to the same datadir.
- train.sh is the main training script. Specify a new modeldir for each train.sh run. The hyperparms.txt file used in training will be saved in modeldir for future reference. 
- At the end, your workingdir will have a single datadir containing multiple BPE'ed versions of the bitext, and multiple modeldir's. You can run tensorboard on all these modeldir's concurrently to compare learning curves.

There are many options in Sockeye, as shown in the help message:
 
```bash
conda activate sockeye3
python -m sockeye.train --help
```

Note that sockeye-recipe hyperameters have the same name as sockeye hyperparameters, except that sockeye-recipe hyperparameters replace the hyphen with underscore (e.g. --num-embed in sockeye becomes $num_embed in sockeye-recipes). The most commonly-used hyperparameters are exposed in sockeye-recipes hpm files, but it is possible to define anything. 


## Hyperparameter Optimization (AutoML) 

Hyperparameters are important to the model building process; manual tuning can be laborious. 
There are various scenarios where we may want an automated hyperparameter optimization process: 
(1) Given a new dataset for a new domain or language-pair, one may wish to find a strong baseline through rigorous hyperparameter optimization. 
(2) Given a novel NMT implementation, one may wish to try a broad range of hyperparameters to ensure that the innovation is sufficiently tested. 

The `automl` subdirectory contains code for hyperparameter optimization. The `egs` subdirectory contains some example methods:

* [egs/gridsearch](egs/gridsearch): Grid search is a straightforward (but potentially expensive) brute-force method. The code here illustrates how the repo manages multiple training runs. 

* [egs/asha](egs/asha): Asynchronous Successive Halving Algorithm (ASHA) is a bandit learning method that learns to automatically terminate training runs that appear less promising than others. It is effective for large hyperparameter spaces on large grid setups. 

