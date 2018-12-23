"""
Script to assemble GT superblocks from GT blocks
"""
import json
import time
import math

from cerebellum.data_prep.seg_prep import *
from cerebellum.skeletonize import gen_skeletons

###
# SET PARAMS
resolution = (30, 48, 48)
block_size = 60
affinity_offset = 14
n_blocks = 15
block_ids = range(0,n_blocks)
block_shape = (block_size*n_blocks, 540, 489)
superblock_steps = [1, 2, 4, 8, 12, 16] # save after assembling these many blocks
full_superblock_folder = '/home/srujanm/cerebellum/segs/gt-superblock-full' # location of full GT superblock
###
gt_superblock_type = 'gt48nm-linear-superblock'

total_runtime = 0
for n_blocks in superblock_steps:
    block_time = time.time()
    gt_superblock_name = gt_superblock_type + '-%dblocks'%(n_blocks)
    print "Assembling superblock with %d blocks"%(n_blocks)
    print gt_superblock_name
    gt_superblock_seg = SegPrep(gt_superblock_name, resolution)
    gt_superblock_seg.read(full_superblock_folder+'/seg.h5', 'main',
                            block_lims=((0,n_blocks*block_size),(None,None),(None,None)))
    gt_superblock_seg.read_bboxes(external_path=full_superblock_folder+'/bboxes.json')
    gt_superblock_seg.relabel(use_bboxes=True)
    gt_superblock_seg.write()
    gen_skeletons(gt_superblock_name, overwrite_prev=True)
    block_runtime = time.time()-block_time
    total_runtime += block_runtime
print "Completed assembling superblocks. Total runtime: %f"%(total_runtime)