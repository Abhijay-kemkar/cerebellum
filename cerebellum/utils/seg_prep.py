import os
import numpy as np

from data_io import *
from timing import TimingDecorators

class SegPrep(object):
    """
    Methods to prepare a segmentation for the error correction pipeline
    """
    def __init__(self, name, res):
        """
        Initializes seg prep task, generates .meta file for input segmentation

        Attributes:
            name (str): folder name for all outputs associated with this segmentation
            resolution (float, float, float): physical resolution in order Z, Y, X in nm
            data (3D ndarray): segmentation data
        """
        self.name = name
        self.resolution = res
        self.shape = None
        self.data = None

        create_folder('./meta/')
        meta = open("./meta/" + name +'.meta', "w")
        meta.write("# resolution in nm\n")
        meta.write("%dx%dx%d\n"%(res[0], res[1], res[2]))
        meta.close()
        create_folder('./logs/')
        log = open("./logs/" + name +'.log', "w")
        log.write("created .meta file\n\n")
        log.close()

    def read(self, filename, datasetname, dsmpl=(1,1,1), block_lims=3*((None,None),)):
        """
        Args:
            filename (str)
            datasetname (str): name of group
            dsmpl (int, int, int): downsampling factor along each axis
            block_lims (tuple of 3 tuples, each with 2 ints): extents of array to read in along each axis
        """
        self.data = read3d_h5(filename, datasetname, dsmpl=dsmpl, block_lims=block_lims)
        self.shape = list(self.data.shape)
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("read from parent file\n")
        log.write(os.path.abspath(filename)+'\n')
        log.write("downsampling factor from parent file\n")
        log.write("%dx%dx%d\n"%(dsmpl[0], dsmpl[1], dsmpl[2]))
        log.write("slices read from parent file\n")
        log.write("[%d:%d, %d:%d, %d:%d]\n"%(block_lims[0][0], block_lims[0][1], block_lims[1][0], block_lims[1][1], block_lims[2][0], block_lims[2][1]))
        log.close()

    #TODO - add timing decorator
    #@TimingDecorators.print_runtime
    def relabel(self):
        """
        Relabels the segmentation data such that max ID = # objects - 1
        """
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("relabeled flag\n")
        seg_ids = np.unique(self.data)
        n_ids = len(seg_ids)
        max_id = np.max(seg_ids)
        if max_id == n_ids-1:
            log.write("False\n")
            log.close()
        else:
            missing_ids = np.sort(np.array(list(set(range(max_id+1)).difference(set(seg_ids)))))
            id_map = seg_ids
            for i in range(len(missing_ids)):
                if i==len(missing_ids)-1:
                    ids_to_correct = range(missing_ids[i]+1, max_id+1)
                else:
                    ids_to_correct = range(missing_ids[i]+1, missing_ids[i+1])
                for j in ids_to_correct:
                    self.data[self.data==j] = j-(i+1) #TODO (Jeff): speed this up using object-wise bounding boxes
                    id_map[j-(i+1)] = j
            log.write("True\n")
            log.close()
            create_folder('./segs/')
            create_folder('./segs/' + self.name)
            idout = "./segs/" + self.name + "/relabeling-map.npy"
            np.save(idout, id_map)

    def padzeros(self, nzeros, axis):
        """
        Pads zeros at end along chosen axis
        Args:
            nzeros (int, int, int): number of zeros along each dimension
            axis (int): axis along which zeros must be appended
        """
        padding = np.zeros((nzeros[0], nzeros[1], nzeros[2]),dtype=np.uint32)
        self.data = np.append(self.data, padding, axis=axis)
        self.shape[axis] += nzeros[1]
        meta = open("./logs/" + self.name +'.log', "a+")
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("zero-padded flag\n")
        log.close()

    def write(self):
        create_folder('./segs/')
        fout = './segs/' + self.name + "/seg.h5"
        writeh5(fout, 'main', self.data, compression="gzip", chunks=(1,self.data.shape[1],self.data.shape[2]))
        meta = open("./meta/" + self.name +'.meta', "a+")
        meta.write("# grid size\n")
        meta.write("%dx%dx%d\n"%(self.shape[0], self.shape[1], self.shape[2]))
        meta.close()
        log = open("./logs/" + self.name +'.log', "a+")
        log.write("saved prepared segmentation\n\n")
        log.close()