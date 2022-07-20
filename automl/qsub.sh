# submit train
qsub -l mem_free=200G,h_rt=10:00:00,num_proc=10,gpu=1 -q gpu.q -o {0}.log.o -e {0}.log.e -N {1} ../scripts/train.sh -p {2} -e sockeye3
# submit evaluation
qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=00:30:00 -o {0}.log.o -e {0}.log.e -N {1} -j y ../scripts/translate.sh -p {2} -i ../egs/ted/multitarget-ted/en-zh/tok/ted_dev_en-zh.tok.zh -o {3}/ted_dev_en-zh.tok.en.1best -e sockeye3