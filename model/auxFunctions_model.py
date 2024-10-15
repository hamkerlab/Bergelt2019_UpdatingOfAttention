# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:16:59 2018

@author: juschu


auxiliary functions for model scripts:
 - print progress information (during simulation and creation of connections)
 - store / load dictionaries with h5py
   see https://codereview.stackexchange.com/questions/120802/
       recursively-save-python-dictionaries-to-hdf5-files-using-h5py
"""


#################
#### imports ####
#################
import time
import sys
import h5py
import numpy as np


################
#### output ####
################
class ProgressOutput:
    '''
    print some progress information
    - during simulation:
        print elapsed time and estimated time needed for simulation
    - during creation of connections:
        print status of creating projection (current timestep and estimated elapsed time,
        number and percentage of created connections)
    '''

    def __init__(self, *args):
        '''
        init output class

        args: connection -- string of projection name
              pre, post  -- populations that are connected
        (only for progress of creation of connections)
        '''

        self.startTime = time.time()

        if args:
            # only for progress of creation of connections
            self.connection = args[0]
            self.prename = args[1].name
            self.postname = args[2].name

            print "Create Connection {0} -> {1} with pattern {2}".format(self.prename, self.postname,
                                                                         self.connection)


    def print_sim(self, value, maxvalue):
        '''
        print elapsed and estimated time of simulation

        params: value    -- current timestep
                maxvalue -- duration of simulation
        '''

        endtime = time.time() - self.startTime
        timestr = "%3d:%02d" % (endtime / 60, endtime % 60)

        progress = value*100.0/(maxvalue)

        if progress > 0:
            estimatedtime = endtime / progress * 100
            estimatedtimestr = "%3d:%02d" % (estimatedtime / 60, estimatedtime % 60)
        else:
            estimatedtimestr = "--:--"

        print "{0: <7} |{1: <7}".format(
            timestr, estimatedtimestr)

    def print_conn(self, value, maxvalue, connectioncreated, full):
        '''
        print status of creating projection

        params: value             -- current step of creating projection
                maxvalue          -- number of steps of creating projection
                connectioncreated -- number of created connections
                full              -- print full information or simply status of progress
        '''

        if full:
            endtime = time.time() - self.startTime
            timestr = "%3d:%02d" % (endtime / 60, endtime % 60)

            progress = value*100.0/(maxvalue)

            if progress > 0:
                estimatedtime = endtime / progress * 100
                estimatedtimestr = "%3d:%02d" % (estimatedtime / 60, estimatedtime % 60)
            else:
                estimatedtimestr = "--:--"

            print "{0} | {1} -> {2} | {3:.4f}% | {4: <7} | {5: <7} | {6: >15}".format(
                self.connection, self.prename, self.postname,
                value*100.0/(maxvalue), timestr, estimatedtimestr, connectioncreated)

        else:
            sys.stdout.write('.')
            sys.stdout.flush()
            if value == maxvalue:
                sys.stdout.writelines(" done.\n")


###########################################
#### store / load dictionary with hdf5 ####
###########################################
def save_dict_to_hdf5(dic, filename):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Save a dictionary whose contents are only strings, np.float64, np.int64,
    np.ndarray, and other dictionaries following this structure
    to an HDF5 file. These are the sorts of dictionaries that are meant
    to be produced by the ReportInterface__to_dict__() method.
    """

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def load_dict_from_hdf5(filename):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Load a dictionary whose contents are only strings, floats, ints,
    numpy arrays, and other dictionaries following this structure
    from an HDF5 file. These dictionaries can then be used to reconstruct
    ReportInterface subclass instances using the
    ReportInterface.__from_dict__() method.
    """

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """

    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py._hl.files.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        #print(key,item)
        key = str(key)
        if isinstance(item, list):
            item = np.array(item)
            #print(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        # save strings, numpy.int64, and numpy.float64 types
        if isinstance(item, (np.int64, np.float64, str, np.float, float, np.float32, int)):
            #print( 'here' )
            h5file[path + key] = item
            if not h5file[path + key].value == item:
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save numpy arrays
        elif isinstance(item, np.ndarray):
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype('|S9')
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key].value, item):
                raise ValueError('The data representation in the HDF5 file does not match the original dict.')
        # save dictionaries
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        # other types cannot be saved and will result in an error
        else:
            #print(item)
            raise ValueError('Cannot save %s type.' % type(item))

def recursively_load_dict_contents_from_group(h5file, path):
    """
    see https://codereview.stackexchange.com/questions/120802/
                recursively-save-python-dictionaries-to-hdf5-files-using-h5py

    Load contents of an HDF5 group. If further groups are encountered,
    treat them like dicts and continue to load them recursively.
    """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
