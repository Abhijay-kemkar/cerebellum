import matplotlib.pyplot as plt

from voxel_methods import *
from cerebellum.utils.data_io import *

class VoxEval(object):
    """Methods for voxel based evaluation of segmentations"""
    def __init__(self, gt_name, pred_name):
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
        self.pred = read3d_h5('./segs/' + pred_name + '/seg.h5', 'main')
        self.pred[self.gt==0] = 0 # zero voxels that are unlabeled in GT
        self.results_folder = './err-analysis/' + pred_name
        self.missing_objs = None
        self.iou_results = None
        self.delta_vis = None

        try:
            assert all([(d1==d2) for d1, d2 in zip(self.gt.shape, self.pred.shape)])
        except:
            print "GT and pred arrays have different dimensions"
        create_folder('./err-analysis/')
        create_folder(self.results_folder)
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("initialized voxel based evaluation against " + gt_name + "\n\n")
        log.close()

    def find_misses(self):
        """finds GT objects missing in pred, removes them for further evaluation methods"""
        self.missing_objs = pred_fails(self.gt, self.pred, do_save=True, 
                                        write_file=self.results_folder+'/missing-objs')
        print "Found %d GT objects completely missing"%(self.missing_objs.size)
        for i in self.missing_objs.tolist():
            self.gt[self.gt==i] = 0
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("missing object count\n")
        log.write("%d\n"%(self.missing_objs.size))
        log.close()

    def find_ious(self, print_thresh=0.7, show_hist=False):
        """finds IoU scores of all objects in segmentation"""
        self.iou_results = iou_rank(self.gt, self.pred, 
                                     Ngt=None, do_save=True, 
                                     write_file=self.results_folder+"/iou_results")
        iou_scores = self.iou_results[2,:]
        print_count = np.count_nonzero(iou_scores>print_thresh)
        print "found %d objects with IoU score below %f"%(print_count, print_thresh)
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
        """runs deltaVI calculation"""
        self.delta_vis = vi_rank(self.gt, self.pred, self.iou_results, 
                                 iou_max=iou_max, do_save=True, 
                                 write_file=self.results_folder+"/vi_scores")
        gt_vi = self.delta_vis[0,:]
        deltaVI = self.delta_vis[1:,:]
        n_segs = min(15, deltaVI.shape[0])
        ind = np.arange(n_segs)
        width = 0.5
        fig, ax = plt.subplots(1)
        psplit = ax.bar(ind, deltaVI[0,:n_segs], width, color='#d62728')
        pmerge = ax.bar(ind, deltaVI[1,:n_segs], width, bottom=deltaVI[:n_segs,0])
        plt.ylabel('delta_VI')
        plt.xlabel('GT id of segment')
        plt.title('Objects with highest delta_VI')
        plt.xticks(ind, gt_vi[:n_segs].astype(int))
        plt.legend((psplit[0], pmerge[0]), ('delta_VI split', 'delta_VI merge'))
        fig.set_size_inches(16, 6)
        plt.savefig(self.results_folder+'/deltaVI.png', bbox_inches="tight")
        plt.close()
        log = open("./logs/" + self.pred_name +'.log', "a+")
        log.write("delta_VI calculation complete for objects with IoU score below %f\n\n"%(iou_max))
        log.close()