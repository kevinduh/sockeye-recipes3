#!/bin/bash
#
# Preprocess train and validation data with BPE

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: preprocess-bpe.sh -p hyperparams.txt -e ENV_NAME"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

while getopts ":h?p:e:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi


###########################################
# (0) Hyperparameter settings 
## source hyperparams.txt to get text files and #symbols for BPE
check_file_exists $HYP_FILE
source $HYP_FILE
source $rootdir/install/path.sh
conda activate $ENV_NAME

## standard settings, need not modify
character_coverage=0.9995

# (1) Save new BPE'ed data and vocab file in datadir 
mkdir -p $datadir


###########################################
# (2) BPE on source side
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on source and creating vocabulary: $bpe_vocab_src"
$rootdir/scripts/sentpiece.py train --input=$train_tok.$src --vocab_size=$bpe_symbols_src --model_type=bpe --model_prefix=$bpe_vocab_src --character_coverage=${character_coverage} 

echo `date '+%Y-%m-%d %H:%M:%S'` "- Applying BPE, creating: ${train_bpe_src}, ${valid_bpe_src}" 
$rootdir/scripts/sentpiece.py encode --input=$train_tok.$src --model=$bpe_vocab_src.model > $train_bpe_src
$rootdir/scripts/sentpiece.py encode --input=$valid_tok.$src --model=$bpe_vocab_src.model > $valid_bpe_src

###########################################
# (3) BPE on target side
echo `date '+%Y-%m-%d %H:%M:%S'` "- Learning BPE on target and creating vocabulary: $bpe_vocab_trg"
$rootdir/scripts/sentpiece.py train --input=$train_tok.$trg --vocab_size=$bpe_symbols_trg --model_type=bpe --model_prefix=$bpe_vocab_trg --character_coverage=${character_coverage}

echo `date '+%Y-%m-%d %H:%M:%S'` "- Applying BPE, creating: ${train_bpe_trg}, ${valid_bpe_trg}" 
$rootdir/scripts/sentpiece.py encode --input=$train_tok.$trg --model=$bpe_vocab_trg.model > $train_bpe_trg
$rootdir/scripts/sentpiece.py encode --input=$valid_tok.$trg --model=$bpe_vocab_trg.model > $valid_bpe_trg

echo `date '+%Y-%m-%d %H:%M:%S'` "- Done with preprocess-bpe.sh"
