#!/bin/bash
#
# Train a Neural Machine Translation model using Sockeye

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: train.sh -p hyperparams.txt -e ENV_NAME [-d DEVICE]"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}


while getopts ":h?p:e:d:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    d) DEVICE=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi

###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
source $HYP_FILE

# system-specific cpu/gpu and conda settings (may need to modify for different grids)
source $rootdir/install/path.sh
conda activate $ENV_NAME
source $rootdir/install/get-device.sh $DEVICE ""

###########################################
# (1) Book-keeping
mkdir -p $modeldir
cp $HYP_FILE $modeldir/hyperparams.txt
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "$0 $@" >> $modeldir/cmdline.log
echo "$devicelog" >> $modeldir/cmdline.log

# this script generates $trainargs which contains standard arguments for sockeye.train
source $rootdir/scripts/arguments-for-train.sh -p $HYP_FILE


###########################################
# (2) Train the model (this may take a while) 
python -m sockeye.train -s $train_bpe_src \
                        -t $train_bpe_trg \
                        -vs $valid_bpe_src \
                        -vt $valid_bpe_trg \
                        $trainargs


##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End training: $datenow on $(hostname)" >> $modeldir/cmdline.log
echo "===========================================" >> $modeldir/cmdline.log
