"""
Script to assemble superblocks from previously linked blocks
"""
import json
import time
import math

from cerebellum.data_prep.seg_prep import *

###
# SET PARAMS
resolution = (120,128,128)
affinity_offset =  14 # affinity offset along z-axis
block_size = 60 # but n_slices is 15
n_slices = 15
wz_thresh = 500
n_blocks = 42
block_ids = range(0,n_blocks)
block_shape = (n_slices*n_blocks-1, 230, 696)
superblock_steps = [n_blocks] # save after assembling these many blocks
iou_thresh = 0.3 # IMPORTANT
chain_name = "tracked-locked-to-0-iou-%.2f"%(iou_thresh)
###

# assemble linearly generated blockchain linked to 0 block
save_folder = './superblocks/'
create_folder(save_folder)
save_fname = "waterz_128nm_th%d-"%(wz_thresh) + chain_name
total_runtime = 0
superblock = np.zeros(block_shape, dtype=np.uint32) 
for block_id in block_ids:
    block_time = time.time()
    zz = block_id*block_size + affinity_offset
    block_name = "waterz_%04d_128nm_th%d"%(zz, wz_thresh)
    print "Appending the %dth block: "%(block_id) + block_name
    block = SegPrep(block_name, resolution)
    if block_id==0: block.read_internal(stage="tracked")
    else: block.read_internal(stage=chain_name)
    print "Finished reading block"
    # if block_id==n_blocks-1:
    #     superblock[block_id*n_slices:(block_id+1)*n_slices-1,
    #                :,:] = block.data
    # else:
    superblock[block_id*n_slices:(block_id+1)*n_slices,
               :,:] = block.data
    block_runtime = time.time()-block_time
    print "Finished operations with block %d in %f seconds"%(block_id, block_runtime)
    total_runtime += block_runtime
    if block_id+1 in superblock_steps:
        print "Saving superblock with %d blocks"%(block_id+1)
        writeh5(save_folder+save_fname+ '-%dblocks.h5'%(block_id+1), 'main', 
                superblock[:(block_id+1)*n_slices,:,:], compression='gzip')
print "Completed assembling superblocks. Total runtime: %f"%(total_runtime)