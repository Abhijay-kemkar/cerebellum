"""
Script to link IDs across blocks
"""
from cerebellum.data_prep.seg_prep import *
from cerebellum.block_chain.block_lock import *

import json

###
# SET PARAMS
resolution = (120,128,128)
affinity_offset =  14 # affinity offset along z-axis
block_size = 60 # but n_slices is 15
wz_thresh = 500
n_blocks = 42
sblock_ids = range(0,n_blocks-1)
iou_thresh = 0.3 # IMPORTANT
chain_name = "tracked-locked-to-0-iou-%.2f"%(iou_thresh)
###

# link blocks in linear fashion
start_time = time.time()
for sblock_id in sblock_ids:
    tblock_id = sblock_id + 1
    # load sblock
    zz_sb = sblock_id*block_size + affinity_offset
    sblock_name = "waterz_%04d_128nm_th%d"%(zz_sb, wz_thresh)
    sblock = SegPrep(sblock_name, resolution)
    if sblock_id==0: sblock.read_internal(stage="tracked")
    else: sblock.read_internal(stage=chain_name)
    sblock_seg = sblock.data
    sbbox_dict = sblock.bbox_dict
    # load tblock
    zz_tb = tblock_id*block_size + affinity_offset
    tblock_name = "waterz_%04d_128nm_th%d"%(zz_tb, wz_thresh)
    tblock = SegPrep(tblock_name, resolution)
    tblock.read_internal(stage="tracked")
    tblock_seg = tblock.data
    tbbox_dict = tblock.bbox_dict
    print "Linking block %d and %d"%(sblock_id, tblock_id)
    tblock_locked = block_lock(sblock_seg, tblock_seg, iou_thresh=iou_thresh, 
                               tbbox_dict=tbbox_dict)
    tblock.data = tblock_locked
    tblock.write(stage=chain_name)
print "Completed block-chain. Total runtime: %f"%(time.time()-start_time)