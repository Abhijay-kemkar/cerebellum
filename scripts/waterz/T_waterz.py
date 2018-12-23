"""
Script to run waterz on an affinity block from the new affinity networks and generate a segmentation
"""
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
aff_loc = '/n/coxfs01/donglai/srujan/vol1_test_'
aff_type = '49Ktrain'
# WATERZ PARAMS
aff_lo = 0.05
aff_hi = 0.995
mf = 'aff85_his256' # waterz histogram params
wz_thresh= [0.5] # waterz threshold
seg_prefix = '/n/coxfs01/donglai/srujan/segs/seg_' + aff_type + '_' # base result location
###

affname = aff_loc + aff_type + '/aff_%04d.h5'%(i*n_slices)
load_time = time.time()
print '1. Loading affinity: ', affname
aff = np.array(h5py.File(affname, 'r')['main'], dtype=np.float32)/255.0 # assumes affinity is stored as uint8
#aff = np.array(h5py.File(affname, 'r')['main'], dtype=np.float32) # assumes affinity is stored as float between 0 and 1
load_time = time.time()-load_time
print 'Finished loading in %d s. Affinity shape: '%(load_time), aff.shape
print 'Memory usage: %d bytes'%(aff.nbytes)

wz_time = time.time()
print '3. Running waterz with %s and threshold %.2f'%(mf, wz_thresh[0])
seg_prefix = seg_prefix + '%04d_'%(i*n_slices)
print 'Will save result to: ', seg_prefix
waterz.waterz(aff, wz_thresh, merge_function=mf, 
              output_prefix=seg_prefix, return_seg=False,
              fragments_mask=None,
              aff_threshold=[aff_lo,aff_hi], m_history=True)
wz_time = time.time()-wz_time
print 'Finished running waterz in %d s'%(wz_time)

total_time = load_time+wz_time
print 'Completed waterz processing in time: %d s'%(total_time)
