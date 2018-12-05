import numpy as np
# ibex imports
from ibex.transforms.seg2seg import DownsampleMapping
from ibex.skeletonization.generate_skeletons import TopologicalThinning, FindEndpointVectors, FindEdges

from cerebellum.utils.data_io import *
from cerebellum.data_prep.seg_prep import *

def check_ids(seg):
    """
    checks if all IDs in segmentation are consecutive
    Args:
        seg (ndarray): segmentation data
    """
    seg_ids = np.unique(seg)
    max_id = np.max(seg_ids)
    n_ids = len(seg_ids)
    try:
        assert max_id == n_ids-1
    except:
        missing_ids = np.sort(np.array(list(set(range(max_id+1)).difference(set(seg_ids)))))
        print "Error! Labels in segmentation are not consecutive. %d IDs are missing"%(len(missing_ids))
        print missing_ids

def gen_skeletons(seg_name, resolution, stage=None, dsmpl_res=(80,80,80), overwrite_prev=False):
    """
    Function to skeletonize a segmentation
    Args:
        seg_name (str): name of prepped segmentation
        stage (str): stage of segmentation in pipeline for skeletonization, eg. "filtered", "tracked", etc.
        dsmpl_res (int, int, int): resolution to downsample segmentation to before skeletonziation in nm
    """
    seg = SegPrep(seg_name, resolution)
    seg.read_internal(stage=stage)
    if stage is not None:
        if overwrite_prev is False:
            print "I currently do not support saving different skeleton files for each stage. "+\
                    "You can choose to overwrite previous skeletons if you wish."
        else:
            print "Warning: overwriting previously generated skeletons, possibly from a different " +\
                    "stage of %s"%(seg_name)
    check_ids(seg.data)
    seg_for_skel = seg.data.astype(np.int64)
    create_folder('./skeletons/')
    create_folder('./skeletons/' + seg_name)
    if overwrite_prev:
        print "Starting skeletonization of " + seg_name
        DownsampleMapping(seg_name, seg_for_skel, output_resolution=dsmpl_res)
        TopologicalThinning(seg_name, seg_for_skel, skeleton_resolution=dsmpl_res)
        FindEndpointVectors(seg_name, skeleton_resolution=dsmpl_res)
        FindEdges(seg_name, skeleton_resolution=dsmpl_res)
        log = open("./logs/" + seg_name +'.log', "a+")
        log.write("completed skeletonization procedure; skeleton count\n")
        log.write("%d\n"%(np.max(seg.data)+1))
        log.close()
    else:
        print "Skeletonization on " + seg_name + " was already completed"