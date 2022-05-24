## Hyperparameter file templates for sockeye-recipes

These files should be self-explanatory. The general naming convention is:

* First letter: {t = Transformer}
* Second letter: {s = small model, m = mid-sized model, l = large model}
* Third numeral: Just an identifier

So, for example:

* ts1 = Transformer model, relatively small size, id=1. These have around 50M parameters (exact number depends on vocabulary size)
* tm1 = Transformer model, mid-sized. These have alround 80M+ parameters. id=1 corresponds to Tranformer Base in original paper.
* tl1 = Transformer model, large. These have around 250M+ parameters. id=1 corresponds to Tranformer Big in original paper

Additionally, some deep encoder shallow decoder models are available. Naming convention is:

* l1-e10d2 = follows most of the hyperparameter of tl1, but changed encoder to 10 layers and decoder to 2 layers.

These hyperparameter file templates are meant to be used as starting points. The best hyperparameter setting will of course depend on the task. 

---

The hyperparameter file templates are designed with a minimal set of commonly-used hyperparameters. It should suffice for mose use cases. But if extra Sockeye arguments are needed for training, there are two ways to do this. 

First, extra arguments may be passed to `scripts/train.sh` with the `-r` flag, e.g.:

```
scripts/train.sh -p ts1.hpm -e sockeye3 -r "--transformer-dropout-attention 0.3:0.3 --max-seconds 3600000" 
```

This will run training with those extra hyperparameter arguments, and also append an `EXTRA_ARGS` variable to the end of the `hyperparams.txt` file in the model directory for documentation. Extra arguments can be added repeatedly and future argument values overwrite the old ones, like so:

```
scripts/train.sh -p ts1/hyperparams.txt -e sockeye3 -r "--transformer-activation-type swish1:swish1"
```

The example's result is seen in `ts1-extra-args.hpm-template`: note `EXTRA_ARGS` have been appended; if there are duplicates, only the final one matters. 

The second way to add extra arguments is to directly specify the `EXTRA_ARGS` in the hpm file and run `train.sh` without the `-r` flag. Both ways can be used together. 