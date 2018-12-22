"""
Script to link IDs across blocks
"""
from cerebellum.data_prep.seg_prep import *
from cerebellum.block_chain.block_lock import *

import json

###
# SET PARAMS
resolution = (30, 48, 48)
block_size = 60
affinity_offset = 14
wz_thresh = 0.5
n_blocks = 16
sblock_ids = range(0,n_blocks-1)
iou_thresh = 0.5 # IMPORTANT: threshold to link ids
chain_name = "locked-to-0" # prefix for linked segmentation file to write to disk
###

# link blocks in linear fashion
start_time = time.time()
for sblock_id in sblock_ids:
    tblock_id = sblock_id + 1
    # load source block
    zz_sb = sblock_id*block_size + affinity_offset
    sblock_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz_sb)
    sblock = SegPrep(sblock_name, resolution)
    if sblock_id==0: sblock.read_internal(stage="filtered")
    else: sblock.read_internal(stage=chain_name)
    sblock.read_bboxes() # Warning! If objects are relabeled, load relabeled-bboxes.json
    sblock_seg = sblock.data
    sbbox_dict = sblock.bbox_dict
    # load target block
    zz_tb = tblock_id*block_size + affinity_offset
    tblock_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz_tb)
    tblock = SegPrep(tblock_name, resolution)
    tblock.read_internal(stage="filtered")
    tblock.read_bboxes() # Warning! If objects are relabeled, load relabeled-bboxes.json
    tblock_seg = tblock.data
    tbbox_dict = tblock.bbox_dict
    # link IDs of target block to IDs in source block
    print "Linking block %d and %d"%(sblock_id, tblock_id)
    tblock_locked = block_lock(sblock_seg, tblock_seg, iou_thresh=iou_thresh, 
                               tbbox_dict=tbbox_dict)
    tblock.data = tblock_locked
    tblock.write(stage=chain_name)
print "Completed block-chain. Total runtime: %f"%(time.time()-start_time)
