#!/usr/bin/env python3

"""Simple wrapper for sentencepiece train and encode
"""

import sys
import argparse
import sentencepiece as spm


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        spm.SentencePieceTrainer.Train(" ".join(sys.argv[2:]))

    elif sys.argv[1] == 'encode':

        p = argparse.ArgumentParser()
        p.add_argument("mode")
        p.add_argument("--model", required=True)
        p.add_argument("--input", required=True)
        args = p.parse_args()

        sp = spm.SentencePieceProcessor(model_file=args.model)
        with open(args.input, 'r') as INFILE:
            for line in INFILE:
                newline = sp.encode(line, out_type=str)
                print(" ".join(newline))

    elif sys.argv[1] == 'debpe':
        
        p = argparse.ArgumentParser()
        p.add_argument('mode')
        p.add_argument('input', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
        args = p.parse_args()
        for line in args.input:
            print(line.replace(' ', '').replace('▁', ' ').strip(), flush=True)

    else:
        print("Usage: sentpiece.py mode [options...], where mode={train, encode, debpe}")
        exit(1)
