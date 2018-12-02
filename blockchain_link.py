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
n_blocks = 15
sblock_ids = range(0,n_blocks-1)
###

# link blocks in linear fashion
start_time = time.time()
for sblock_id in sblock_ids:
    tblock_id = sblock_id + 1
    # load sblock
    zz_sb = sblock_id*block_size + affinity_offset
    sblock_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz_sb)
    sblock = SegPrep(sblock_name, resolution)
    if sblock_id==0: sblock.read_internal(stage="filtered")
    else: sblock.read_internal(stage="locked-to-0")
    sblock.read_bboxes() # Warning! If objects are relabeled, load relabeled-bboxes.json
    sblock_seg = sblock.data
    sbbox_dict = sblock.bbox_dict
    # load tblock
    zz_tb = tblock_id*block_size + affinity_offset
    tblock_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz_tb)
    tblock = SegPrep(tblock_name, resolution)
    tblock.read_internal(stage="filtered")
    tblock.read_bboxes() # Warning! If objects are relabeled, load relabeled-bboxes.json
    tblock_seg = tblock.data
    tbbox_dict = tblock.bbox_dict
    print "Linking block %d and %d"%(sblock_id, tblock_id)
    tblock_locked = block_lock(sblock_seg, tblock_seg, iou_thresh=0.5, 
                               tbbox_dict=tbbox_dict)
    tblock.data = tblock_locked
    tblock.write(stage="locked-to-0")
print "Completed block-chain. Total runtime: %f"%(time.time()-start_time)