import numpy as np
import argparse
import h5py
import time
from em_segLib.seg_eval import CremiEvaluate

def IoU_gen():
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
	    seg_gt_id = seg_gt_id[:200]
	    seg_gt_cc = seg_gt_cc[:200]
	    ind = seg_gt_id[seg_gt_cc>4000]
	    num = len(ind)
	    iou = np.zeros(num)
	    a=open('/home/srujanm/snemi/analysis_results/gt_analysis','w')
	    for i in range(num):
	        tmp_id, tmp_cc = np.unique(seg_p[seg_gt==ind[i]], return_counts=True)
	        #print len(tmp_cc)
	        tmp_sid = np.argsort(-tmp_cc) # descend
	        iou[i] = float(max(tmp_cc))/len(np.union1d(np.ravel_multi_index(np.where(seg_gt==ind[i]),seg_gt.shape), np.ravel_multi_index(np.where(seg_p==tmp_id[np.argmax(tmp_cc)]),seg_p.shape))) 
	        numV = float(sum(tmp_cc))
	        msg = '%d,%d,%.2f' %(ind[i],numV,iou[i])
	        for j in range(min(5,len(tmp_cc))):
	            msg+=',%d,%.2f'%(tmp_id[tmp_sid[j]],tmp_cc[tmp_sid[j]]/numV)
	        print msg 
	        a.write(msg+'\n')
	    a.close()
	return IoU

def gt_evaluation(IoU_set,pred,gt,seg_gt,top=20,original_rand=None,maxT=3600):
    import em_segLib
    import time
    #from em_segLib.seg_dist import DilateData
    IoU_set = IoU_set[(-IoU_set[:,1]).argsort()]
    #print IoU_set
    #seg_gt = DilateData(gt,6)
    gt_id,area,IoU = IoU_set[:,0],IoU_set[:,1],IoU_set[:,2]
    o_rand = orignal_rand if original_rand is not None else CremiEvaluate(pred.astype(np.int), seg_gt.astype(np.int))
    start = time.time()
    print 'current VI-merge: %1.5f' %(o_rand[1])
    max_id = np.max(gt) + 1
    diff_list = []
    for i in range(len(gt_id)):
        if area[i] < 5000 or time.time() - start > maxT:
            break
        if IoU[i] > 0.7:
            continue
        tpred = pred.copy()
        tpred[gt==gt_id[i]] = max_id
        rand_score = CremiEvaluate(tpred.astype(np.int), seg_gt.astype(np.int))[1]
        print 'Time remaining:%5d'%(maxT-time.time()+start)
        print 'Label: %9d generate delta arand %1.5f' %(gt_id[i],o_rand[1]-rand_score)
        diff_list.append([gt_id[i],o_rand[1]-rand_score])
        del tpred
    diff_list = np.array(diff_list)
    diff_list = diff_list[(-diff_list[:,1]).argsort()]
    return diff_list

if __name__="__main__":
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

	gt = np.array(h5py.File('/home/srujanm/snemi/zhuhd15_scripts/label.h5','r')['main'])
	pred = np.array(h5py.File('/home/srujanm/snemi/zhuhd15_scripts/tmp_result/withaff85_his256_0.2.h5','r')['main'])

	eval_final = gt_evaluation(IoU,pred.copy(),gt.copy(),gt.copy(),maxT=3600)
	print eval_final
	np.save('/home/srujanm/snemi/analysis_results/delta',eval_final)