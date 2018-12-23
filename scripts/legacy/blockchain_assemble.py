"""
Script to assemble superblocks from previously linked blocks
"""
import json
import time
import math

from cerebellum.data_prep.seg_prep import *

###
# SET PARAMS
resolution = (30, 48, 48)
block_size = 60
affinity_offset = 14
wz_thresh = 0.5
n_blocks = 16
block_ids = range(0,n_blocks)
block_shape = (block_size*n_blocks, 540, 489)
superblock_steps = [1, 2, 4, 8, 12, 16] # save after assembling these many blocks
iou_thresh = 0.4 # IMPORTANT
chain_name = "locked-to-0"#"locked-to-0-iou-%.2f"%(iou_thresh)
###

# assemble linearly generated blockchain linked to 0 block
save_folder = './superblocks/'
create_folder(save_folder)
#save_fname = 'waterz%.2f-48nm-crop2gt-linear-superblock-iou-%.2f'%(wz_thresh, iou_thresh)
save_fname = 'waterz%.2f-48nm-crop2gt-linear-superblock'%(wz_thresh)
total_runtime = 0
superblock = np.zeros(block_shape, dtype=np.uint32) 
for block_id in block_ids:
    block_time = time.time()
    zz = block_id*block_size + affinity_offset
    block_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz)
    print "Appending the %dth block: "%(block_id) + block_name
    block = SegPrep(block_name, resolution)
    if block_id==0: block.read_internal(stage="filtered")
    else: block.read_internal(stage=chain_name)
    print "Finished reading block"
    superblock[block_id*block_size:(block_id+1)*block_size,
               :,:] = block.data
    block_runtime = time.time()-block_time
    print "Finished operations with block %d in %f seconds"%(block_id, block_runtime)
    total_runtime += block_runtime
    if block_id+1 in superblock_steps:
        print "Saving superblock with %d blocks"%(block_id+1)
        writeh5(save_folder+save_fname+ '-%dblocks.h5'%(block_id+1), 'main', 
                superblock[:(block_id+1)*block_size,:,:], compression='gzip')
print "Completed assembling superblocks. Total runtime: %f"%(total_runtime)

# generate downsampled full superblock for visualization
resolution_vis = (96,96,30)
superblock_vis = superblock
print superblock_vis.shape
print "Generating downsampled full superblock for ng visualization"
# zero every 3rd z-slice, so effective z-resolution is 90nm
for z in range(0,superblock_vis.shape[0],3):
    superblock_vis[z,:,:] = 0
writeh5(save_folder+save_fname+'_vis.h5', 'main', superblock_vis, compression="gzip")