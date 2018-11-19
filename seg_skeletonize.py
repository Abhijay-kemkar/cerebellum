"""
Script to skeletonize GT and pred blocks
"""
import json

from cerebellum.skeletonize import gen_skeletons

with open('data_locs.json') as f:
	data_locs = json.load(f)

# TO CHANGE
block_indices = range(1)

# GT blocks
for i in block_indices:
	zz = i*data_locs["block-size"]+data_locs["aff-offset"]
	gt_name = "gt%04d"%(zz)
	print "Starting skeletonization of " + gt_name
	gen_skeletons(gt_name)

# pred blocks
for i in block_indices:
	zz = i*data_locs["block-size"]+data_locs["aff-offset"]
	pred_name = "pred-pf-crop2gt-%04d"%(zz)
	print "Starting skeletonization of " + pred_name
	gen_skeletons(pred_name)