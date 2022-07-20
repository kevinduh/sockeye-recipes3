import os
def check_args(args):
    assert(args.R>=args.r, 
        "R {0} is smaller than r {1}!".format(str(args.R), str(args.r)))
    assert(args.p > 1, 
        "p should be greater than 1!")
    assert(args.G > 0,
        "G should be a positive number!")
    if args.resume_from_ckpt:
        assert(os.path.exists(args.resume_from_ckpt),
        "The path to previous checkpoint does not exist: " + 
        args.resume_from_ckpt)