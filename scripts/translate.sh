#!/bin/bash
#
# Translate an input file with a Sockeye NMT model

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: translate.sh -p hyperparams.txt -i input -o output -e ENV_NAME [-d DEVICE] [-c checkpoint] [-s] [-b batchsize] [-v beamsize]"
  errcho "Input is a source text file to be translated"
  errcho "Output is filename for target translations"
  errcho "ENV_NAME is the sockeye conda environment name"
  errcho "Device is optional and inferred from ENV"
  errcho "Checkpoint is optional and specifies which model checkpoint to use while decoding (-c 00005)"
  errcho "-s is optional and skips BPE processing on input source"
  errcho "Additional decoding flags: -b batchsize -v beamsize"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}

while getopts ":h?p:e:i:o:d:c:sb:v:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
    e) ENV_NAME=$OPTARG
      ;;
    i) INPUT_FILE=$OPTARG
      ;;
    o) OUTPUT_FILE=$OPTARG
      ;;
    d) DEVICE=$OPTARG
      ;;
    c) CHECKPOINT=$OPTARG
      ;;
    s) SKIP_SRC_BPE=1
      ;;
    b) BATCH_SIZE=$OPTARG
      ;;
    v) BEAM_SIZE=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE || -z $ENV_NAME || -z $INPUT_FILE || -z $OUTPUT_FILE ]]; then
    errcho "Missing arguments"
    show_help
    exit 1
fi

if [[ -z $BATCH_SIZE ]]; then
    BATCH_SIZE=1
fi

if [[ -z $BEAM_SIZE ]]; then
    BEAM_SIZE=5
fi

###########################################
# (0) Setup
# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
check_file_exists $INPUT_FILE
source $HYP_FILE

# system-specific cpu/gpu and conda settings (may need to modify for different grids)
source $rootdir/install/path.sh
conda activate $ENV_NAME
source $rootdir/install/get-device.sh $DEVICE ""

# If the checkpoint is provided, add the argument tag
[ -z $CHECKPOINT ] || CHECKPOINT="-c $CHECKPOINT"

###########################################
# (1) Book-keeping
LOG_FILE=${OUTPUT_FILE}.log
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "Start translating: $datenow on $(hostname)" > $LOG_FILE
echo "$0 $@" >> $LOG_FILE
echo "$devicelog" >> $LOG_FILE

###########################################
# (2) Translate!

if [ "$SKIP_SRC_BPE" == 1 ]; then
    echo "Directly translating source input without applying BPE" >> $LOG_FILE
    cmd_preprocess="cat $INPUT_FILE"
else
    echo "Apply BPE to source input before translating" >> $LOG_FILE
    cmd_preprocess="$rootdir/scripts/sentpiece.py encode --input $INPUT_FILE --model $bpe_vocab_src.model "
fi

$cmd_preprocess | \
    python -m sockeye.translate --models $modeldir $device \
    --batch-size $BATCH_SIZE --beam-size $BEAM_SIZE $CHECKPOINT 2>> $LOG_FILE | \
    $rootdir/scripts/sentpiece.py debpe  > $OUTPUT_FILE 

##########################################
datenow=`date '+%Y-%m-%d %H:%M:%S'`
echo "End translating: $datenow on $(hostname)" >> $LOG_FILE
echo "===========================================" >> $LOG_FILE
