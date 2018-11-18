import h5py
import numpy as np
import os

def writeh5(filename, datasetname, array, compression=None, chunks=None):
    """
    Writes np array to hdf5 file as a dataset
    """
    fid = h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, shape=array.shape, compression=compression, chunks=chunks, dtype=array.dtype)
    ds[:] = array
    fid.close()

def read3d_h5(filename, datasetname, dsmpl=(1,1,1), block_lims=3*((None,None),)):
    """
    Reads and returns 3D np array from hdf5 file
    Args:
        filename (str)
        datasetname (str): name of group
        dsmpl (tuple of 3 ints): downsampling factor along each axis
        block_lims (tuple of 3 tuples, each with 2 ints): extents of array to read in along each axis
    Returns:
        array: 3D ndarray
    """
    fid = h5py.File(filename,'r')
    ds = fid[datasetname]
    read_slice = tuple([slice(block_lims[i][0],block_lims[i][1],dsmpl[i]) for i in range(3)])
    array = np.array(ds[read_slice])
    fid.close()
    return array

def create_folder(fpath):
    if not os.path.isdir(fpath):
            os.mkdir(fpath)
    else:
        pass