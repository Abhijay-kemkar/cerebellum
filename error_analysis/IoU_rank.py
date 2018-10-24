import numpy as np
import argparse
import h5py
import time

parser = argparse.ArgumentParser(description='IoU Evalutation')
parser.add_argument('-p','--iou_path', type=str, default='/home/srujanm/snemi/analysis_results/')
parser.add_argument('-f','--iou_file', type=str, default='gt_analysis')
parser.add_argument('-i','--id', type=str, default='B')
args = parser.parse_args()

IoU = []

with open(args.iou_path+args.iou_file,'r') as iou_File:
    for line in open(args.iou_path+args.iou_file):
        line = iou_File.readline().split(',')
        IoU.append([int(t) for t in line[:2]]+[float(line[2])])
IoU = np.array(IoU)
print IoU.shape

gt_id = IoU[:,0]
area = IoU[:,1]
IoU_area = IoU[:,2]
gt = np.array(h5py.File('/home/srujanm/snemi/zhuhd15_scripts/label.h5','r')['main'])
pred = np.array(h5py.File('/home/srujanm/snemi/zhuhd15_scripts/tmp_result/withaff85_his256_0.2.h5','r')['main'])
from em_segLib.seg_eval import adapted_rand,CremiEvaluate#,arand_simple
from em_segLib.seg_dist import DilateData
import em_segLib
#gt_dilate = 

from processing_util import gt_evaluation

eval_final = gt_evaluation(IoU,pred.copy(),gt.copy(),gt.copy(),maxT=3600)
eval_total = eval_final[(-eval_final[:,1]-eval_final[:,2]).argsort()]
print eval_total
eval_split = eval_final[(-eval_final[:,1]).argsort()]
eval_merge = eval_final[(-eval_final[:,2]).argsort()]
np.save('/home/srujanm/snemi/analysis_results/delta_total',eval_total)
np.save('/home/srujanm/snemi/analysis_results/delta_split',eval_split)
np.save('/home/srujanm/snemi/analysis_results/delta_merge',eval_merge)
