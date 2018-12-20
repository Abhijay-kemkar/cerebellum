import h5py
import numpy as np
import time
import argparse
import em_segLib
import waterz

###
# CMD ARGS
parser = argparse.ArgumentParser(description='run waterz agglomeration on affinity blocks')
parser.add_argument('-i', '--block_index', type=int, default=0, help='block index')
args = parser.parse_args()
i = args.block_index
# AFF PARAMS
n_slices = 60 # number of slices per block
aff_loc = '/home/srujanm/cerebellum/data/retrained_vol1/affs_'
aff_type = 'train49K'
# MASK SEG PARAMS
# seg_init_loc = '/home/srujanm/cerebellum/data/vol1/waterz/waterz0.50_pf-align1_1k_crop.h5' # location of initial seg
# dsmpl = (4,16,16)
# bvol_thresh = float(n_slices)/4*40 # cell body volume threshold
#aff_off_init = 14 # affinity offset for initial seg
#upsmpl_init = 6 # x,y resolution upsampling factor for initial seg
# WATERZ PARAMS
aff_lo = 0.05
aff_hi = 0.995
mf = 'aff85_his256' # waterz histogram params
wz_thresh= [0.5] # waterz threshold
seg_prefix = '/home/srujanm/cerebellum/data/retrained_vol1/segs/seg_' + aff_type + '_' # base result location
###

affname = aff_loc + aff_type + '/aff_%04d.h5'%(i*n_slices)
load_time = time.time()
print '1. Loading affinity: ', affname
#aff = np.array(h5py.File(affname, 'r')['main'], dtype=np.float32)/255.0 # assumes affinity is stored as uint8
aff = np.array(h5py.File(affname, 'r')['main'], dtype=np.float32) # assumes affinity is stored as float between 0 and 1
load_time = time.time()-load_time
print 'Finished loading in %d s. Affinity shape: '%(load_time), aff.shape
print 'Memory usage: %d bytes'%(aff.nbytes)

# mask_time = time.time()
# print '2. Generating mask for cell bodies'
# #seg_init_file = seg_init_loc + 'waterz0.50-48nm-crop2gt-%d/filt-dsmpl-seg.h5'%(i*n_slices+aff_off_init)
# seg_init = np.array(h5py.File(seg_init_loc, 'r')['main'][i*n_slices:(i+1)*n_slices]) # initial segmentation
# print 'Loaded initial filtered segmentation of shape', seg_init.shape
# assert seg_init.shape == aff[0,:].shape
# #seg_init_upsmpl = seg_init.repeat(upsmpl, axis=1).repeat(upsmpl, axis=2)
# print 'Downsampling by: ', dsmpl
# seg_dsmpl = seg_init[::dsmpl[0], ::dsmpl[1], ::dsmpl[2]]
# labels, vols = np.unique(seg_dsmpl, return_counts=True)
# body_ids = [label[0] for label in labels[np.argwhere(vols>bvol_thresh)]]
# print "Found %d cell bodies"%(len(body_ids))
# fragments_mask = np.logical_not(np.isin(seg_init, body_ids)) # mask of all objects that are not cell bodies
# del seg_init, seg_dsmpl
# mask_time = time.time()-mask_time
# print 'Finished mask generation in %d s'%(mask_time)
fragments_mask = None
mask_time = 0

wz_time = time.time()
print '3. Running waterz with %s and threshold %.2f'%(mf, wz_thresh[0])
seg_prefix = seg_prefix + '%04d_'%(i*n_slices)
print 'Will save result to: ', seg_prefix
waterz.waterz(aff, wz_thresh, merge_function=mf, 
              output_prefix=seg_prefix, return_seg=False,
              fragments_mask=fragments_mask,
              aff_threshold=[aff_lo,aff_hi], m_history=True)
wz_time = time.time()-wz_time
print 'Finished running waterz in %d s'%(wz_time)

#save_time = time.time()
#segname = seg_file+'%04d.h5'%(i*n_slices)
#print '3. Saving segmentation to %s'%(segname)
#fid = h5py.File(segname,'w')
#ds = fid.create_dataset('main', data=seg, compression="gzip")
#fid.close()
#save_time = time.time()-save_time
#print 'Finished saving in %d s'%(save_time)
save_time = 0

total_time = load_time+mask_time+wz_time+save_time
print 'Completed waterz processing in time: %d s'%(total_time)
