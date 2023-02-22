import os
import sys


def run_script(arg1, arg2, arg3):
     os.system('/home/vsel/ice_drift_pc_ncc/tools/alpha_shapes/alpha_shapes --in {} --out {} --alpha {}'.format(arg1, arg2, arg3))


def make_alpha_shape(f):
    print(f'making alpha shapes for {f}')
    alpha = 5 #km
    in_arg = f
    out_arg = f+'_alpha_shape_'+str(alpha)+'km.txt'
    run_script(in_arg,out_arg, (alpha*1000)**2)


if len(sys.argv) != 2:
    raise Exception('expected file path')

make_alpha_shape(sys.argv[1])