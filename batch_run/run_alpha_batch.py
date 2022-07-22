# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import numpy as np

# def list_directories(path):
#     all = (os.path.join(path, item) for item in os.listdir(path))
#     all_dirs = [item for item in all if os.path.isdir(item)]
#     all_dirs.sort()
#     return all_dirs


def list_files(path):
    all = (os.path.join(path, item) for item in os.listdir(path) if not item.startswith('mask'))
    all_files = [item for item in all if os.path.isfile(item)]
    all_files.sort()
    return all_files


def run_script(arg1, arg2, arg3):
     # os.system(f'python3 /home/len-min/lera/run_scripts/test_script.py {arg1} {arg2} 1 2 3')
     # os.system('python /home/lera/GOIN/Drift/ice_drift_pc_ncc/cc_bm_parallel_pyr_dev.py {arg1} {arg2} 64 4 30')
     os.system('/home/lera/Lenya/alpha_shapes/alpha_shapes --in {} --out {} --alpha {}'.format(arg1, arg2, arg3))


path = '/home/lera/GOIN/Perspektiva/ESS_stam/Calculations/res/2019/for_alpha_сс/'
outdir = '/home/lera/GOIN/Perspektiva/ESS_stam/Calculations/res/2019/out_alpha_сс/'
files_in_dir = list_files(path)

for fpath in files_in_dir:
    basename = os.path.basename(fpath)
    alpha = 5# in km
    in_arg = fpath
    out_arg = outdir + basename[:-4]+'_'+str(alpha)+'km.txt'
    run_script(in_arg,out_arg, (alpha*1000)**2)

# path = '/home/lera/GOIN/Perspektiva/ESS_stam/Calculations/res/2016/for_alpha'
# outdir = '/home/lera/GOIN/Perspektiva/ESS_stam/Calculations/res/2016/out_alpha/'


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     run_script_for_each_dir('/home/len-min/lera/run_scripts/tst_data')
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
