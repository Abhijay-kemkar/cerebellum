from cerebellum.data_prep.seg_prep import *
from cerebellum.error_correction.block_track import *

data_folder = '/home/srujanm/cerebellum/data/pnj/15000/'
resolution = (120,128,128)
affinity_offset =  14 # affinity offset along z-axis
block_size = 60 # but n_slices is 15
wz_thresh = 500
iou_thresh = 0.3

n_blocks = 42
block_ids = range(n_blocks)
for block_id in block_ids:
    zz = block_id*block_size + affinity_offset
    pnj_block_name = "waterz_%04d_128nm_th%d"%(zz, wz_thresh)
    data_file = data_folder + pnj_block_name + '.h5'
    print "Prepping ", data_file
    pnj_block = SegPrep(pnj_block_name, resolution)
    pnj_block.read(data_file, 'main')
    print pnj_block.shape
    pnj_block.write()
    pnj_block.data = block_track(pnj_block.data, iou_thresh, zstride=1)
    pnj_block.write(stage="tracked")