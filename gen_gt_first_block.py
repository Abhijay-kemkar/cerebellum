"""
Script to generate GT blocks of chosen size with good labels
"""

import json
import math

from cerebellum.data_prep.seg_prep import *

###
# SET PARAMS
with open('data_locs.json') as f:
	data_locs = json.load(f)
resolution = (30, 48, 48)
dsmpl = (1,6,6) # from 8 nm
gt8nm_file = "/home/srujanm/cerebellum/data/vol1/gt8nm_pf-align1_1k_crop.h5"
bbox = data_locs["gt"]["8nm-bbox"] # global bbox
block_size = 60
total_slices = h5py.File(gt8nm_file)["main"].shape[0]
assert total_slices==bbox[3]-bbox[0]
###

n_blocks = 1 #total_slices/block_size
print "Starting generation of %d blocks"%(n_blocks)
for i in range(n_blocks):
	print "Generating block %d from"%(i)
	print gt8nm_file
	zz = i*block_size+data_locs["aff-offset"]
	gt_block_name = "gt-48nm-%dslices-%04d"%(block_size, zz)
	gt_block = SegPrep(gt_block_name, resolution)
	block_lims = ((block_size*i, block_size*(i+1)),(None,None),(None,None))
	print block_lims
	gt_block.read(gt8nm_file, "main", dsmpl=dsmpl, block_lims=block_lims)
	print gt_block.shape
	gt_block.gen_bboxes()
	#gt_block.read_bboxes()
	gt_block.relabel(use_bboxes=True, print_labels=False)
	gt_block.write()