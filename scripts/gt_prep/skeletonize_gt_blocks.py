"""
Script to generate skeletons of a GT block
"""
from cerebellum.skeletonize import gen_skeletons
import argparse

###
# CMD ARGS
parser = argparse.ArgumentParser(description='skeletonize a block of ground truth')
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
args = parser.parse_args()
i = args.block_index
###
aff_offset = 14 # affinity offset along z-axis
block_size = 60
n_blocks = 16
resolution = (30,48,48)
###

zz = i*block_size+aff_offset
gt_block_name = "gt-48nm-%dslices-%04d"%(block_size, zz)
gen_skeletons(gt_block_name, resolution, overwrite_prev=True)
