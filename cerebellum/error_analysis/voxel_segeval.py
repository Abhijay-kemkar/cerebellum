import json
import matplotlib.pyplot as plt

from .voxel_methods import *
from cerebellum.utils.data_io import *

class VoxEval(object):
    """High level methods for voxel based evaluation of segmentations"""
    def __init__(self, gt_name, pred_name, stage=None):
        """
        Attributes:
            gt_name (str): name of GT segmentation
            pred_name (str): name of predicted segmentation
            gt (ndarray): GT segmentation data
            pred (ndarray): predicted segmentation data
            results_folder (str)
            Evaluation results:
            missing_objs
            iou_results
            delta_vis
        """
        self.gt_name = gt_name
        self.pred_name = pred_name
        self.gt = read3d_h5('./segs/' + gt_name + '/seg.h5', 'main')
        self.stage = stage
        if stage is None:
            self.pred = read3d_h5('./segs/' + pred_name + '/seg.h5', 'main')
            self.results_folder = './err-analysis/' + pred_name
        else:
            self.pred = read3d_h5('./segs/' + pred_name + '/' + stage + '-seg.h5', 'main')
            self.results_folder = './err-analysis/' + pred_name + '/' + stage
        self.pred[self.gt==0] = 0 # zero voxels that are unlabeled in GT
        self.missing_objs = read_npy(self.results_folder+'/missing-objs.npy')
        self.vi = read_json(self.results_folder+'/vi.json')
        self.iou_results = read_npy(self.results_folder+'/iou_results.npy')
        self.delta_vis = read_npy(self.results_folder+'/vi_scores.npy')
        print("Starting voxel evaluation methods for " + pred_name + " against " + gt_name)
        if self.missing_objs is None:
            if stage is not None:
                create_folder('./err-analysis/' + pred_name)
            create_folder(self.results_folder)
            log = open("./logs/" + self.pred_name +'.log', "a+")
            log.write("initialized voxel based evaluation against " + gt_name + "\n\n")
            log.close()

    def remove_from_gt(self, ids):
        """zeros voxel values of all GT objects listed in ids"""
        for i in ids:
            self.gt[self.gt==i] = 0

    def find_misses(self, thresh_miss=0):
        """finds GT objects missing in pred, removes them for further evaluation methods"""
        self.missing_objs = pred_fails(self.gt, self.pred, thresh_miss=thresh_miss,
                                        do_save=True, 
                                        write_file=self.results_folder+'/missing-objs')
        print("Found %d GT objects completely missing"%(self.missing_objs.size))
        self.remove_from_gt(self.missing_objs.tolist())
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("missing object count\n")
        log.write("%d\n"%(self.missing_objs.size))
        log.close()

    def find_vi(self):
        """finds VI of pred against GT"""
        self.remove_from_gt(self.missing_objs.tolist())
        self.vi = calc_vi(self.gt, self.pred, do_save=True, write_file=self.results_folder+'/vi.json')
        print("VI split, VI merge: %f, %f"%(self.vi[0], self.vi[1]))
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("VI calculation completed\n")
        log.write("%f, %f\n"%(self.vi[0], self.vi[1]))
        log.close()

    def find_ious(self, print_thresh=0.7, show_hist=False):
        """
        finds IoU scores of all objects in segmentation
        Args:
            print_thresh (float): print # objects with IoU below this threshold
            show_hist (bool): flag to show histogram of all IoU scores
        """
        self.remove_from_gt(self.missing_objs.tolist())
        self.iou_results = iou_rank(self.gt, self.pred, 
                                     Ngt=None, do_save=True, 
                                     write_file=self.results_folder+"/iou_results")
        iou_scores = self.iou_results[2,:]
        print_count = np.count_nonzero(iou_scores>print_thresh)
        print("found %d objects with IoU score below %f"%(print_count, print_thresh))
        if show_hist:
            plt.hist(x=iou_scores[1:], bins=10) # remove segment 0
            plt.xlabel('IoU')
            plt.ylabel('Number of segments')
            plt.yscale('log')
            plt.title('IoU scores statistics')
            plt.show()
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("IoU below %f object count\n"%(print_thresh))
        log.write("%d\n"%(print_count))
        log.close()

    def find_delta_vis(self, iou_max=0.7):
        """
        runs deltaVI calculation
        Args:
            iou_max (float): run delta_VI calculation for all segments with IoU score below this threshold
            hist_segs (int): number of segments in delta_VI histogram
        """
        self.remove_from_gt(self.missing_objs.tolist())
        self.delta_vis = vi_rank(self.gt, self.pred, self.iou_results, 
                                 iou_max=iou_max, do_save=True, 
                                 write_file=self.results_folder+"/vi_scores")
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("delta_VI calculation complete for objects with IoU score below %f\n\n"%(iou_max))
        log.close()

    def gen_vi_histograms(self, hist_segs=10):
        """
        Plots histograms of contributions of inidivudal GT Objects to deltaVI
        """
        gt_vi = self.delta_vis[0,:]
        deltaVI = self.delta_vis[1:,:]
        # generate delatVI histograms
        # segments ranked by delta_VI total
        n_segs = min(hist_segs, deltaVI.shape[1])
        ind = np.arange(n_segs)
        width = 0.5
        fig, ax = plt.subplots(1)
        psplit = ax.bar(ind, deltaVI[0,:n_segs], width, color='#d62728')
        pmerge = ax.bar(ind, deltaVI[1,:n_segs], width, bottom=deltaVI[0,:n_segs])
        plt.ylabel('delta_VI')
        plt.xlabel('GT id of segment')
        plt.title('Objects with highest delta_VI')
        plt.xticks(ind, gt_vi[:n_segs].astype(int))
        plt.legend((psplit[0], pmerge[0]), ('delta_VI split', 'delta_VI merge'))
        fig.set_size_inches(16, 6)
        plt.savefig(self.results_folder+'/deltaVI.png', bbox_inches="tight")
        plt.close()
        # segments ranked by delta_VI split
        sort_splits = np.argsort(-deltaVI[0,:])
        gt_vi = gt_vi[sort_splits]
        deltaVI = deltaVI[:,sort_splits]
        fig, ax = plt.subplots(1)
        psplit = ax.bar(ind, deltaVI[0,:n_segs], width, color='#d62728')
        pmerge = ax.bar(ind, deltaVI[1,:n_segs], width, bottom=deltaVI[0,:n_segs])
        plt.ylabel('delta_VI')
        plt.xlabel('GT id of segment')
        plt.title('Objects with highest delta_VI split')
        plt.xticks(ind, gt_vi[:n_segs].astype(int))
        plt.legend((psplit[0], pmerge[0]), ('delta_VI split', 'delta_VI merge'))
        fig.set_size_inches(16, 6)
        plt.savefig(self.results_folder+'/deltaVI_split.png', bbox_inches="tight")
        plt.close()
        # segments ranked by delta_VI merge
        sort_merges = np.argsort(-deltaVI[1,:])
        gt_vi = gt_vi[sort_merges]
        deltaVI = deltaVI[:,sort_merges]
        fig, ax = plt.subplots(1)
        psplit = ax.bar(ind, deltaVI[0,:n_segs], width, color='#d62728')
        pmerge = ax.bar(ind, deltaVI[1,:n_segs], width, bottom=deltaVI[0,:n_segs])
        plt.ylabel('delta_VI')
        plt.xlabel('GT id of segment')
        plt.title('Objects with highest delta_VI merge')
        plt.xticks(ind, gt_vi[:n_segs].astype(int))
        plt.legend((psplit[0], pmerge[0]), ('delta_VI split', 'delta_VI merge'))
        fig.set_size_inches(16, 6)
        plt.savefig(self.results_folder+'/deltaVI_merge.png', bbox_inches="tight")
        plt.close()
        print("Delta VI histogram plots generated and saved")

    def run_fullsuite(self, thresh_miss=0, iou_max=0.7, hist_segs=10, overwrite_prev=False):
        """
        Runs all voxel evaluation methods. If previously run, loads existing results

        Args:
            iou_max (float): run delta_VI calculation for all segments with IoU score below this threshold
            overwrite_prev (bool): overwrite previous results if any
        """
        if self.missing_objs is None or overwrite_prev:
            print("Starting missing object evaluation")
            self.find_misses(thresh_miss=thresh_miss)
        else:
            print("Missing object results loaded")
        if self.vi is None or overwrite_prev:
            print("Starting VI evaluation")
            self.find_vi()
        else:
            print("VI loaded")
        if self.iou_results is None or overwrite_prev:
            print("Starting IoU evaluation")
            self.find_ious(print_thresh=iou_max)
        else:
            print("IoU results loaded")
        if self.delta_vis is None or overwrite_prev:
            print("Starting delta VI evaluation")
            self.find_delta_vis(iou_max=iou_max)
        else:
            print("delta_VI results loaded")
        self.gen_vi_histograms(hist_segs=hist_segs)

    def get_vi(self):
        """returns VI as tuple of two floats: (VI split, VI merge)"""
        try:
            return (self.vi["VI split"], self.vi["VI merge"])
        except:
            print("Error retrieving VI because VI calculation has not been run yet")