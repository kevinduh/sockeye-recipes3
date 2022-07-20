import sys
import os
import random
import shutil
from utils.hpm_file import update_hpm

def main():
    run_dir = sys.argv[1]
    num_hpm_subspace = int(sys.argv[2])

    hpm_space_dir = os.path.join(os.getcwd(), "hpms")
    if not os.path.exists(hpm_space_dir):
        print("The directory to hpm files does not exist: ", \
            os.path.join(os.getcwd(), hpm_space_dir))
        exit()
    hpm_subspace_dir = os.path.join(os.getcwd(), run_dir, "hpms")
    os.makedirs(hpm_subspace_dir, exist_ok=True)
    model_dir = os.path.join(os.getcwd(), run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), run_dir, "job_logs"), exist_ok=True)

    total_num = len(os.listdir(hpm_space_dir))
    if num_hpm_subspace == -1 or num_hpm_subspace >= total_num:
        sub_num = total_num
    else:
        sub_num = num_hpm_subspace

    hpm_subspace = random.sample(os.listdir(hpm_space_dir), sub_num)
    for hpm in hpm_subspace:
        shutil.copyfile(os.path.join(hpm_space_dir, hpm), \
            os.path.join(hpm_subspace_dir, hpm))
        modelname = hpm[:-4]
        modeldir = os.path.join(os.getcwd(), model_dir, modelname)
        update_hpm(os.path.join(hpm_subspace_dir, hpm),
            {'modeldir': modeldir})

if __name__ == "__main__":
    main()