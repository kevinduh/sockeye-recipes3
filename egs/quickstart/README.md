## Quickstart: sockeye-recipes


We will train a model on a very small sample German-English data, just to confirm our installation works. The whole process should take less than 30 minutes. Since the data is so small, you should not expect the model to learn anything.

(1) Download and unpack the data in any directory. Let's use the current directory `$rootdir/egs/quickstart` as working directory. Note that `$rootdir` is defined as the base directory of the sockeye-recipes repository. 

```bash
cd $rootdir/egs/quickstart/
./0_download_data.sh
```

(2) Peruse the hyperparameters file. This is the file that specifies everything about an NMT run. The example hyperparameters file `tiny_transformer.hpm` has been prepared for you

```bash
cat tiny_transformer.hpm
```

Note that `workdir=./` so everything will be placed in the current directory. The relative path `$rootdir=../../` should point to your base directory in the sockeye-recipes repo; this can also be edited to a hardcoded path. 

(3) Preprocess data with BPE segmentation.

Run the `preprocess-bpe.sh` script, which will read the training/validation tokenized bitext (via `train_tok` and `valid_tok`), learn the BPE subword units (the number of which is specified by `bpe_symbols_src` and `bpe_symbols_trg`) and save the BPE'd text in `datadir=$workdir/data-bpe`

```bash
bash path/to/sockeye-recipes/scripts/preprocess-bpe.sh -p tiny_transformer.hpm -e sockeye3
```

In practice, you can just run in the current directory:
```bash
bash ../../scripts/preprocess-bpe.sh tiny_transformer.hpm
  2018-06-01 15:37:41 - Learning BPE on source and creating vocabulary: .//data-bpe//train.bpe-4000.de.sentpiece
  2018-06-01 15:37:54 - Applying BPE, creating: .//data-bpe//train.bpe-4000.de, .//data-bpe//valid.bpe-4000.de
  2018-06-01 15:37:58 - Learning BPE on target and creating vocabulary: .//data-bpe//train.bpe-4000.en.sentpiece
  2018-06-01 15:38:07 - Applying BPE, creating: .//data-bpe//train.bpe-4000.en, .//data-bpe//valid.bpe-4000.en
  2018-06-01 15:38:10 - Done with preprocess-bpe.sh
```

This is a standard way (though not the only way) to handle large vocabulary in NMT. Currently sockeye-recipes assumes BPE segmentation before training. The preprocess-bpe.sh script takes a hyperparams file as input and preprocesses accordingly. To get a flavor of BPE segmentation results (train.en is original, train.bpe-4000.en is BPE'ed):

```bash
head -3 sample-de-en/train.en data-bpe/train.bpe-4000.en
```

(4a) Now, we can train the NMT model. Generally, the invocation is:

```bash
bash path/to/sockeye-recipes/scripts/train -p HYPERPARAMETER_FILE -e SOCKEYE_ENVIRONMENT
```
The hyperparameter file specifies the model architecture and training data, while the Sockeye Conda Environment specifies the actual code.

First, let's try the CPU version:

```bash
bash ../../scripts/train-textformat.sh -p tiny_transformer.hpm -e sockeye3 -d cpu
```

The `train-textformat.sh` script starts of the training process on parallel text input (there is another version of `train.sh` that first prepares the text data into another format for efficiency purposes). The `-p` flag indicates the hyperparameter file, and the `-e` flag indicates the sockeye environment you use to use. GPU mode is default when available but `-d cpu` forces it to use CPU. The model and all training info are saved in `modeldir=tiny_transformer/`. 

(4b) Second, let's try the GPU version. This assumes your machine has NVIDIA GPUs. First, we modify the `$modeldir` in the hyperparameter, to keep the training information separate. Next we run the same train-textformat.sh script but telling it to use the GPU environment:

```bash
sed "s|tiny_transformer|tiny_transformer_gpu|" tiny_transformer.hpm > tiny_transformer_gpu.hpm
bash ../../scripts/train-textformat.sh -p tiny_transformer_gpu.hpm -e sockeye3
```

Various sockeye-recipe scripts call install/get-device.sh to determine which device to use. If you are having trouble with the GPU run, check this script to see if it matches your computer system. 

Alternatively, all these commands can also be used in conjunction with the Sun/Univa Grid Engine, e.g.:

```
sed "s|tiny_transformer|tiny_transformer_gpu|" tiny_transformer.hpm > tiny_transformer_gpu.hpm

qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=24:00:00,num_proc=1 -j y ../../scripts/train-textformat.sh -p tiny_transformer_gpu.hpm -e sockeye3

(for CLSP grid)
qsub -S /bin/bash -V -cwd  -l gpu=1,h_rt=24:00:00,num_proc=1 -j y ../../scripts/train-textformat.sh -p tiny_transformer_gpu.hpm -e sockeye3
```

For Multi-GPU training, you can set `-l gpu=1` to a larger number `-l gpu=2`; the script relies on CUDA_VISIBLE_DEVICES to pick the free GPU cards. It is strongly recommended that CUDA_VISIBLE_DEVICES is set in your system; if not, the script will pick a single free GPU based on nvidia-smi (but this is not guaranteed to be safe in a multi-user enivornment).


(5) Finally, we can translate new test sets with `translate.sh`:

Generally, the invocation is:
```bash
bash path/to/sockeye-recipes/scripts/translate.sh -i INPUT_SOURCE_FILE -o OUTPUT_TRANSLATION_FILE -p HYPERPARAMETER_FILE -e ENV
```

Local CPU version: 
```bash
bash ../../scripts/translate.sh -i sample-de-en/valid.de -o tiny_transformer/valid.en.1best -p tiny_transformer.hpm -e sockeye3 -d cpu
```

Qsub GPU version: 
```bash
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:50:00 -j y ../../scripts/translate.sh -i sample-de-en/valid.de -o tiny_transformer_gpu/valid.en.1best -p tiny_transformer_gpu.hpm -e sockeye3
```

This `translate.sh` script will find the model from hyperparams file. Then it runs BPE on the input (which is assumed to be tokenized in the same way as train_tok and valid_tok), translates the result, runs de-BPE and saves in output.

(6) To visualize the learning curve, you can use tensorboard:

```bash
conda activate sockeye3
tensorboard --logdir ./
```

Then follow the instructions, e.g. pointing your browser to http://localhost:6006 . Note that not all features of Google's tensorboard is implemented in this DMLC MXNet port, but at least you can currently visualize perplexity curves and a few other useful things. 


All results are stored in the `$modeldir`. The ones of interest:

```bash
ls tiny_transformer/*
 ... 
 tiny_transformer/cmdline.log --> log of the sockeye-recipes script invocation     
 tiny_transformer/hyperparams.txt --> a backup copy of the hpm file (should be same as tiny_transformer.hpm)
 tiny_transformer/log --> log of sockeye training
 tiny_transformer/metrics --> records perplexity, time, etc. at each checkpoint
 tiny_transformer/params.00000 --> saved model at each checkpoint (e.g. 0)
 tiny_transformer/params.00002 --> saved model at each checkpoint (e.g. 2)
 tiny_transformer/params.best --> points to best model according to validation set
 tiny_transformer/tensorboard/ --> event logs for tensorboard visualization
 tiny_transformer/vocab.{src,trg}.0.json --> vocabulary file generated by sockeye
```
