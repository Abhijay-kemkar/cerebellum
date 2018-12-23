from cerebellum.data_prep.seg_prep import *
from cerebellum.error_correction.block_track import *

import argparse
import json

parser = argparse.ArgumentParser(description='prep of tracked PF segmentation')                                         
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
parser.add_argument('-t', '--iou_thresh', type=float, default=0, help='IoU threshold')  
args = parser.parse_args()                                           
block_id = args.block_index

####
# initial data params
with open('data_locs.json') as f:
    data_locs = json.load(f)
resolution = (30, 48, 48)
bbox = data_locs["gt"]["8nm-bbox"] # global bbox
affinity_offset = data_locs["aff-offset"] # affinity offset along z-axis
block_size = 60
wz_thresh = 0.5
filter_method = "dsmpl"
# tracking params
iou_thresh = args.iou_thresh
zstride = 1
####

zz = block_id*block_size + affinity_offset
block_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz)
block = SegPrep(block_name, resolution)
block.read_internal(stage="filt-"+filter_method)
block.data = block_track(block.data, iou_thresh, zstride=1)
block.write(stage="tracked-iou-%.2f"%iou_thresh)