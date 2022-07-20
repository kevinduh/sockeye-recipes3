#!/usr/bin/env python3

def readfile(hpm):
    with open(hpm) as f:
        return f.readlines()

def get_valid_tok(hpm):
    lines = readfile(hpm)
    for line in lines:
        if line.startswith("trg"):
            suffix = line.split('=')[-1].strip()
        if line.startswith("valid_tok"):
            prefix = line.split('=')[-1].strip()
    return prefix + "." + suffix

def update_hpm(hpm, dict):
    '''
    Update hpm file with new (key, value) pair.
    '''

    lines = readfile(hpm)
    ks = []
    with open(hpm, 'w') as f:
        for l in lines:
            newl = l.strip() + "\n"
            for key, value in dict.items():
                if l.startswith(key):
                    newl = key + "=" + str(value) + "\n"
                    ks.append(key)
                    break
            f.write(newl)

        rk = list(set(dict.keys()) - set(ks))
        for key in rk:
            newl = key + "=" + str(dict[key]) + "\n"
            f.write(newl)

