# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys

def list_directories(path):
    all = (os.path.join(path, item) for item in os.listdir(path))
    all_dirs = [item for item in all if os.path.isdir(item)]
    all_dirs.sort()
    return all_dirs


def list_files(path):
    all = (
        os.path.join(path, item) for item in os.listdir(path) 
        if not item.startswith('mask') and item.endswith('tiff')
    )
    all_files = [item for item in all if os.path.isfile(item)]
    all_files.sort()
    return all_files


def run_script(arg1, arg2):
     # os.system(f'python3 /home/len-min/lera/run_scripts/test_script.py {arg1} {arg2} 1 2 3')
     # os.system('python /home/lera/GOIN/Drift/ice_drift_pc_ncc/cc_bm_parallel_pyr_dev.py {arg1} {arg2} 64 4 30')
     os.system('python /home/vsel/ice_drift_pc_ncc/cc_bm_parallel_pyr_dev.py {} {} 24 10 40'.format(arg1, arg2))


def run_script_for_each_dir(path):
    for dir_path in list_directories(path):
        files_in_dir = list_files(dir_path)
        print (files_in_dir)
        if len(files_in_dir) != 2:
            print('Expected to find 2 files in d {dir_path}')
            break
        print (files_in_dir[0], files_in_dir[1])
        run_script(files_in_dir[0], files_in_dir[1])

if len(sys.argv) != 2:
    raise Exception('Expected path to folder with pairs')

path = sys.argv[1]
run_script_for_each_dir(path)

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     run_script_for_each_dir('/home/len-min/lera/run_scripts/tst_data')
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
