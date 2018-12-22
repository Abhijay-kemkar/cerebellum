"""
Script to filter fibers from all blocks of predicted segmentation
"""

from cerebellum.data_prep.seg_prep import *

wz_threshes = [0.5]

###
# SET PARAMS
resolution = (30, 48, 48)
aff_offset = 14 # affinity offset along z-axis
block_size = 60
n_blocks = 16
filter_method = "dsmpl"
filter_params = {"dsmpl": (4,3,3),
                 "bvol-thresh": float(block_size)/4*40}
###

for wz_id, wz_thresh in enumerate(wz_threshes):
	for i in range(n_blocks):
		# load and prepare pred
		print "Filtering %dth block of waterz: %.2f"%(i, wz_thresh)
		zz = i*block_size + aff_offset
		seg_block_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz)
		seg_block = SegPrep(seg_block_name, resolution)
		seg_block.read_internal()
		print seg_block.shape
		# Assuming bboxes are generated, proceed to filter
		seg_block.read_bboxes()
		seg_block.find_fiber_ids(method=filter_method, params=filter_params) # see function for more details on setting non-default filter method and params
		seg_block.filter_fibers() # see function for more details on setting non-default filter method and params
		#seg_block.relabel(use_bboxes=True)
		seg_block.write(stage="filt-"+filter_method) # this saves your filtered segmentation to the ./segs/<seg_block_name>/ folder
