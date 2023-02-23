#!/bin/bash
# Install optional 3rd party tools

set -e

CONDA_HOME=$HOME/.conda/envs

function errcho() {
  >&2 echo $1
}

function show_help() {
  errcho "Install optional 3rd party tools"
  errcho "usage: install_3rdparty.sh [-h] -e ENV_NAME"
  errcho ""
}

function check_dir_exists() {
  if [ ! -d $1 ]; then
    errcho "FATAL: Could not find directory $1"
    exit 1
  fi
}

while getopts ":h?e:" opt; do
  case "$opt" in
    h|\?)
      show_help
      exit 0
      ;;
    e) ENV_NAME=$OPTARG
      ;;
  esac
done

if [[ -z $ENV_NAME ]]; then
  errcho "Missing arguments"
  show_help
  exit 1
fi


source ./install/path.sh

# 1. setup python virtual environment 
source activate $ENV_NAME

pip3 install interpret
pip3 install kaleido
pip3 install jenkspy

