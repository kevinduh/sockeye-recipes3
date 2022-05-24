#!/usr/bin/env python3

import sys
import yaml
import itertools

hpm_space_yaml = sys.argv[1]


def bpe_cmd(h, lang, symbols, train_bpe, valid_bpe):
    cmd = f"{h['rootdir']}/scripts/sentpiece.py train --input={h['train_tok']}.{lang} --vocab_size={symbols} --model_type=bpe --model_prefix={train_bpe}.sentpiece --character_coverage=0.9995; "
    cmd += f"{h['rootdir']}/scripts/sentpiece.py encode --input={h['train_tok']}.{lang} --model={train_bpe}.sentpiece.model > {train_bpe}; "
    cmd += f"{h['rootdir']}/scripts/sentpiece.py encode --input={h['valid_tok']}.{lang} --model={train_bpe}.sentpiece.model > {valid_bpe}"
    return cmd


def prepare_data_cmd(h, train_bpe_src, train_bpe_trg, bpe_symbols_src, bpe_symbols_trg):
    prepare_data=f"{train_bpe_src}.{h['src']}{bpe_symbols_src}-{h['trg']}{bpe_symbols_trg}.prepared_data"
    cmd = f"python -m sockeye.prepare_data -s {train_bpe_src} -t {train_bpe_trg} -o {prepare_data}"
    return cmd


def data_combinations(h):
    prepfile='prep.sh'
    P = open(prepfile, 'w')

    for bpe_symbols_src in h['bpe_symbols_src']:
        train_bpe_src=f"{h['workdir']}/data-bpe/train.bpe-{bpe_symbols_src}.{h['src']}"
        valid_bpe_src=f"{h['workdir']}/data-bpe/valid.bpe-{bpe_symbols_src}.{h['src']}"
        print(bpe_cmd(h, h['src'], bpe_symbols_src, train_bpe_src, valid_bpe_src), file=P)

    for bpe_symbols_trg in h['bpe_symbols_trg']:
        train_bpe_trg=f"{h['workdir']}/data-bpe/train.bpe-{bpe_symbols_trg}.{h['trg']}"
        valid_bpe_trg=f"{h['workdir']}/data-bpe/valid.bpe-{bpe_symbols_trg}.{h['trg']}"
        print(bpe_cmd(h, h['trg'], bpe_symbols_trg, train_bpe_trg, valid_bpe_trg), file=P)

    for bpe_symbols_src in h['bpe_symbols_src']:
        train_bpe_src=f"{h['workdir']}/data-bpe/train.bpe-{bpe_symbols_src}.{h['src']}"
        for bpe_symbols_trg in h['bpe_symbols_trg']:
            train_bpe_trg=f"{h['workdir']}/data-bpe/train.bpe-{bpe_symbols_trg}.{h['trg']}"
            print(prepare_data_cmd(h, train_bpe_src, train_bpe_trg, bpe_symbols_src, bpe_symbols_trg), file=P)

    P.close()
    print("Step1: run the following to prepare the data - /bin/sh prep.sh")

    
def dict_product(**kwargs):
    keys = kwargs.keys()
    vals = []
    for v in kwargs.values():
        if type(v) is list:
            vals.append(v)
        else:
            vals.append([v])
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def hpm_combinations(h):
    for i,hpm in enumerate(dict_product(**h), start=1):
        modeldir=f"{h['workdir']}/{str(i)}"
        write_hpm(hpm, modeldir)
    print(f"Step2: {i} hpm files generated in {h['workdir']}")

    print("Step3: run the following to train...")
    print("qsub -S /bin/bash -V -cwd -q gpu.q -l gpu=1,h_rt=12:00:00,num_proc=1 -j y -t 1:%d -tc 5 sge-train.sh"%i)

def write_hpm(hpm, modeldir):
    hpm['train_bpe_src']=f"{hpm['workdir']}/data-bpe/train.bpe-{hpm['bpe_symbols_src']}.{hpm['src']}"
    hpm['valid_bpe_src']=f"{hpm['workdir']}/data-bpe/valid.bpe-{hpm['bpe_symbols_src']}.{hpm['src']}"
    hpm['train_bpe_trg']=f"{hpm['workdir']}/data-bpe/train.bpe-{hpm['bpe_symbols_trg']}.{hpm['trg']}"
    hpm['valid_bpe_trg']=f"{hpm['workdir']}/data-bpe/valid.bpe-{hpm['bpe_symbols_trg']}.{hpm['trg']}"
    hpm['bpe_vocab_src']=f"{hpm['train_bpe_src']}.sentpiece"
    hpm['bpe_vocab_trg']=f"{hpm['train_bpe_trg']}.sentpiece"
    hpm['modeldir']=modeldir

    with open("%s.hpm"%modeldir, 'w') as HPM_FILE:
        for k,v in hpm.items():
            HPM_FILE.write("%s=%s\n"%(k,v))



with open(hpm_space_yaml) as HPM_SPACE_YAML:
    h = yaml.safe_load(HPM_SPACE_YAML)


data_combinations(h)
hpm_combinations(h)

