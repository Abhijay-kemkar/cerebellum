import os, sys
import numpy as np
import h5py
from T_util import writeh5
import argparse


parser = argparse.ArgumentParser(description='Evaluation for Waterz Result')
#parser.add_argument('-ver','--version',type=str)
#parser.add_argument('-pa','--params',type=str)
args = parser.parse_args()
parser.add_argument('-lt','--loss_type', type=int, default=0, help='Loss type, 0 for IoU, 1 for Myelin')
parser.add_argument('-gt','--ground_truth', type=str, default='/home/srujanm/snemi/zhuhd15_scripts/label.h5')
parser.add_argument('-dir','--pred_dir',type=str,default='/home/srujanm/snemi/zhuhd15_scripts/tmp_result/')
parser.add_argument('-p','--prediction', type=str, default='withaff85_his256_0.2.h5') 
parser.add_argument('-v','--validation', type=int, default=1)
args = parser.parse_args()
#opt = sys.argv[1]
#version = args.version
#params = args.params
#print version,params

if args.loss_type==0: # IoU
    # load data
    #seg_gt = np.array(h5py.File(args.ground_truth)['main'])[14:-14,287:1450,612:1501]#[14:-14,332:1451,276:1465]#[14:-14,542:1484,522:1501]
    #print args.pred_dir+args.prediction
    #seg_p = np.array(h5py.File(args.pred_dir+args.prediction)['main'])
    seg_gt = np.array(h5py.File(args.ground_truth,'r')['main'])
    seg_p = np.array(h5py.File(args.pred_dir+args.prediction,'r')['main'])
    print seg_gt.shape,seg_p.shape
    
    # do count
    seg_gt_id, seg_gt_cc = np.unique(seg_gt, return_counts=True)
    seg_gt_id = np.delete(seg_gt_id,0) # 0:background
    seg_gt_cc = np.delete(seg_gt_cc,0)
    sid = np.argsort(-seg_gt_cc) # descend
    seg_gt_id = seg_gt_id[sid]
    seg_gt_cc = seg_gt_cc[sid]
    
    # find big seg only
    #ind = seg_gt_id[seg_gt_cc>4000]
    seg_gt_id = seg_gt_id[:200] # look at only the 200 biggest GT objects
    seg_gt_cc = seg_gt_cc[:200]
    ind = seg_gt_id[seg_gt_cc>4000] # IDs of all GT segments with >4000 voxels in descending order of volume
    num = len(ind)
    iou = np.zeros(num)
    a=open('/home/srujanm/snemi/analysis_results/gt_analysis','w')
    for i in range(num):    # iterate over GT labels from largest to smallest in volume
        tmp_id, tmp_cc = np.unique(seg_p[seg_gt==ind[i]], return_counts=True)   # find IDs of all predicted objects intersecting with this GT object
        #print len(tmp_cc)
        tmp_sid = np.argsort(-tmp_cc) # sort objects in decreasing order of intersection with this GT object
        # IoU = 'I'ntersection of best overlapped predicted object 'o'ver its 'U'nion with GT object
        iou[i] = float(max(tmp_cc))/len(np.union1d(np.ravel_multi_index(np.where(seg_gt==ind[i]),seg_gt.shape), np.ravel_multi_index(np.where(seg_p==tmp_id[np.argmax(tmp_cc)]),seg_p.shape))) 
        numV = float(sum(tmp_cc)) # num of voxels in GT object 
        # print <GT index>, <voxel count>, <max IoU with pred object>, <pred objects with top 5 intersections and their contribution to this GT object>
        msg = '%d,%d,%.2f' %(ind[i],numV,iou[i])
        for j in range(min(5,len(tmp_cc))):
            msg+=',%d,%.2f'%(tmp_id[tmp_sid[j]],tmp_cc[tmp_sid[j]]/numV)
        print msg 
        a.write(msg+'\n')
    a.close()
