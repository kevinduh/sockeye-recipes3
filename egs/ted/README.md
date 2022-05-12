## Multitarget TED Talks 

This recipe builds systems from the TED Talks corpus at:
http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/ .
There are many languages available (~20), so we can try many different language-pairs. 

### 1. Setup

First, download the data. It's about 575MB compressed and 1.8GB uncompressed.
```bash
sh ./0_download_data.sh
```

Then, setup the task for the language you are interested.
Let's do Chinese (zh) for now.  
The following command creates a new working directory (zh-en) 
and populates it with several hyperparameter files 

```bash
sh ./1_setup_task.sh zh
cd zh-en
ls
```

You should see files like `ts1.hpm` which is one of the hyperparameter files we will run with. Further, the checkpoint interval is 4000 updates and all model information will be saved in ./ts1.

### 2. Preprocessing and Training

First, make sure we are in the correct working directory (`$rootdir/egs/ted/zh-en`). All hyperparameter files and instructions below assume we are in `$rootdir/egs/ted/zh-en`, where `$rootdir` is the location of the sockeye-recipes installation. 

Now, we can preprocess the tokenized training and dev data using BPE.
```bash
../../../scripts/preprocess-bpe.sh -p ts1.hpm -e sockeye3
```

The resulting BPE vocabulary file (for English) is: `data-bpe/train.bpe-30000.en.sentpiece.vocab` and the segmented training file is: `data-bpe/train.bpe-30000.en`. For Chinese, replace `en` by `zh`. These are the files we train on. 

To train, we will use qsub and gpu (On a Tesla V100, this should take about 6 hours):

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=1 -j y ../../../scripts/train-textformat.sh -p ts1.hpm -e sockeye3
```


### 3. Evaluation

Again, make sure we are in the correct working directory (`$rootdir/egs/ted/zh-en`). The test set we want to translate is `../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.zh`. We translate it using ts1 via qsub on gpu (this should take 10 minutes or less):

```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:30:00 -j y ../../../scripts/translate.sh -p ts1.hpm -i ../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.zh -o ts1/ted_test1_en-zh.tok.en.1best -e sockeye3
```

Note you can also pass to `translate.sh` some options `-b batchsize(default:1)` and `-v beamsize(default:5)` to speed up the decoding time.

Alternatively, to translate using CPU:

```bash
qsub -S /bin/bash -V -cwd -q all.q -l h_rt=00:30:00 -j y ../../../scripts/translate.sh -p ts1.hpm -i ../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.zh -o ts1/ted_test1_en-zh.tok.en.1best -e sockeye3 -d cpu
```

When this is finished, we have the translations in `ts1/ted_test1_en-zh.tok.en.1best`. We can now compute the BLEU score by:

```bash
conda activate sockeye3
sacrebleu --tokenize none ../multitarget-ted/en-zh/tok/ted_test1_en-zh.tok.en < ts1/ted_test1_en-zh.tok.en.1best
```

This should give a BLEU score of around 16-17.


### Benchmark Results (Old results from sockeye-recipes2. Need to be updated)

The test set BLEU scores of various tasks are:

 task | ts1   | tb2   | tb3   | tm1   | tm2   | b2-e6d2 |
  --- | ---   | ---   | ---   | ---   | ---   | ---     |
ar-en | 28.75 | 29.33 | 28.16 | 29.16 | 28.74 | 28.72   |
bg-en | 36.84 | 36.07 | 35.17 | 36.88 | 36.16 | 34.95   |
cs-en | 27.82 | 28.19 | 25.57 | 27.65 | 26.38 | 25.70   |
de-en | 33.75 | 33.04 | 31.61 | 33.50 | 33.10 | 32.62   |
fa-en | 22.59 | 22.31 | 20.92 | 22.41 | 21.79 | 21.16   |
fr-en | 35.81 | 35.53 | 34.53 | 35.81 | 35.53 | 34.84   |
he-en | 35.02 | 34.43 | 33.30 | 34.82 | 35.04 | 33.87   |
hu-en | 22.49 | 22.16 | 20.13 | 22.53 | 21.13 | 20.76   |
id-en | 27.92 | 28.03 | 27.11 | 26.29 | 27.44 | 26.86   |
ja-en | 12.47 | 11.87 | 10.66 | 11.80 | 12.19 | 11.81   |
ko-en | 16.16 | 16.03 | 14.35 | 15.81 | 15.83 | 15.69   |
pl-en | 24.66 | 24.28 | 23.28 | 23.96 | 24.05 | 23.33   |
pt-en | 42.40 | 41.74 | 40.93 | 41.99 | 41.80 | 40.95   |
ro-en | 36.27 | 35.69 | 34.60 | 36.04 | 36.17 | 34.74   |
ru-en | 24.49 | 23.66 | 23.17 | 24.03 | 23.88 | 22.55   |
tr-en | 24.12 | 23.03 | 21.90 | 24.26 | 22.43 | 23.04   |
uk-en | 19.30 | 20.06 | 18.77 | 17.57 | 19.13 | 19.09   |
vi-en | 26.38 | 25.79 | 24.63 | 25.66 | 25.57 | 24.73   |
zh-en | 17.01 | 16.30 | 15.70 | 16.73 | 16.31 | 15.93   |
