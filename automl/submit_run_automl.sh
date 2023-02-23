#!/bin/sh

rootdir=/exp/xzhang/sockeye-recipes3/
rundir=$rootdir/egs/asha/space1/run1

python $rootdir/automl/run_asha.py \
                    -r 1 \
                    -u 1 \
                    -R 6\
                    -p 2 \
                    -G 4 \
                    --timer-interval 90 \
                    --workdir $rootdir/egs/asha/space1/run1 \
                    --job-log-dir $rundir/job_logs \
                    --ckpt $rundir/ckpt.json \
                    #--multi-objective \
                    #--resume-from-ckpt $rundir/ckpt.json 
