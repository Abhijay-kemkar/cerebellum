import json
from cerebellum.skeletonize import gen_skeletons

###
with open('data_locs.json') as f:
	data_locs = json.load(f)
aff_offset = data_locs["aff-offset"] # affinity offset along z-axis
block_size = 60
n_blocks = 16
###

for i in range(n_blocks):
	zz = i*block_size+data_locs["aff-offset"]
	gt_block_name = "gt-48nm-%dslices-%04d"%(block_size, zz)
	gen_skeletons(gt_block_name, overwrite_prev=True)