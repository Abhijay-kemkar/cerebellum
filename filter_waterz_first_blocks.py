from cerebellum.data_prep.seg_prep import *


wz_threshes = [0.5]

###
# SET PARAMS
with open('data_locs.json') as f:
	data_locs = json.load(f)
resolution = (30, 48, 48)
bbox = data_locs["gt"]["8nm-bbox"] # global bbox
aff_offset = data_locs["aff-offset"] # affinity offset along z-axis
block_size = 60
###

for wz_id, wz_thresh in enumerate(wz_threshes):
	i = 0 # block index - iterable if chosen
	# load and prepare pred
	print "Preparing 0th block for waterz: %.2f"%(wz_thresh)
	zz = i*block_size + aff_offset
	seg_block_name = "waterz%.2f-48nm-crop2gt-%04d"%(wz_thresh, zz)
	seg_block = SegPrep(seg_block_name, resolution)
	seg_block.read_internal()
	print seg_block.shape
	# seg_block.write()
	# STOP HERE FOR UNFILTERED SEGMENTATION
	# Generate bboxes for fiber filtering
	# seg_block.gen_bboxes() # Note: Longest step in pipeline
	# Assuming bboxes are generated, proceed to filter
	seg_block.read_bboxes()
	filter_params = {"vol-thresh":block_size*20*20, "len-thresh":0.3*block_size}
	seg_block.find_fiber_ids(params=filter_params) # assumes bboxes are generated already
	seg_block.filter_fibers() # see function for more details on setting non-default filter method and params
	#seg_block.relabel(use_bboxes=True)
	seg_block.write(stage="filtered") # this saves your filtered segmentation to the ./segs/<seg_block_name>/ folder