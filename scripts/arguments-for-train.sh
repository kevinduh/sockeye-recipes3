#!/bin/bash
#
# Standard arguments for training scripts

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Usage: std-arg-train.sh -p hyperparams.txt"
  errcho ""
}

function check_file_exists() {
  if [ ! -f $1 ]; then
    errcho "FATAL: Could not find file $1"
    exit 1
  fi
}


while getopts ":h?p:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    p) HYP_FILE=$OPTARG
      ;;
  esac
done

if [[ -z $HYP_FILE ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi

# source hyperparams.txt to get text files and all training hyperparameters
check_file_exists $HYP_FILE
source $HYP_FILE

trainargs="--encoder $encoder \
           --decoder $decoder \
           --num-layers $num_layers \
           --transformer-model-size $transformer_model_size \
           --transformer-attention-heads $transformer_attention_heads \
           --transformer-feed-forward-num-hidden $transformer_feed_forward_num_hidden \
           --weight-tying trg_softmax \
           --optimized-metric $optimized_metric \
           --embed-dropout $embed_dropout \
           --label-smoothing $label_smoothing \
           --batch-size $batch_size \
           --batch-type word \
           --update-interval $update_interval \
           --initial-learning-rate $initial_learning_rate \
           --seed $seed $device \
           --amp \
           --checkpoint-interval $checkpoint_interval \
           --max-num-checkpoint-not-improved $max_num_checkpoint_not_improved \
           --keep-last-params $keep_last_params \
           --decode-and-evaluate $decode_and_evaluate \
           -o $modeldir"


