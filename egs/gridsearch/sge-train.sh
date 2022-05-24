#!/usr/bin/env bash

# run by:
# qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=1 -j y -t 1:10 -tc 5 sge-train.sh

echo $SGE_TASK_ID
../../scripts/train.sh -p models/${SGE_TASK_ID}.hpm -e sockeye3
