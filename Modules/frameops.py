"""
frameops.py

Read, write and manipulate frame image files.

DPX code from https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc (Jack Doerner)

"""

import numpy as np
import scipy.misc as misc
import os
import copy

import Modules.dpx as dpx

IMAGETYPES = [ ".jpg", ".png", ".dpx"]

# Look in folder_path for all the files that are of one of the IMAGETYPES,
# and return a sorted list of lists containing the absolute paths to those files. So if
# there are only png files, the result will be a list containing a single list, but if
# there are jpg and dpx files, there will be two sublists, one for each type.
#
# If deep is True, look at all the subdirectories as well.

def image_files(folder_path,deep=False):

    file_paths = []
    file_lists = []

    for (root,dirs,files) in os.walk(folder_path):
        file_paths.extend([os.path.join(root,f) for f in files])
        if not deep:
            break

    if len(file_paths) == 0:
        return file_lists

    for ext in IMAGETYPES:
        ext_list = [f for f in file_paths if os.path.splitext(f)[1] == ext]
        if len(ext_list) > 0:
            file_lists.append(sorted(ext_list))

    return file_lists

# Keep a copy of the last meta information read

last_meta = None

# Read an image file and return numpy array of pixel values. Extends scipy.misc.imread
# with support for 10-bit DPX files. Always returns RGB images, with pixel values
# normalized to 0..1 range (inclusive).
#
# Note that the dpx library routines already handle (de-)normalization, so we only
# have to handle that for operations we hand off to scipy.misc.

def imread(file_path):

    global last_meta

    file_type = os.path.splitext(file_path)[1]

    if file_type == ".dpx":
        with open(file_path, 'rb') as f:
            meta = dpx.readDPXMetaData(f)
            if meta is None:
                img = None
            else:
                img = dpx.readDPXImageData(f, meta)
                last_meta = copy.deepcopy(meta)
    else:
        img = misc.imread(file_path, mode='RGB')
        img = img.astype('float32') / 255.0

    return img

# Write a numpy array to an image file. Extends scipy.misc.imsave
# with support for 10-bit DPX files. Expects the input array to be
# normalized to 0..1 range (inclusive). If DPX is being saved, then
# if meta information is None, the meta information of the Last
# image loaded will be used.

def imsave(file_path,img,meta=None):

    global last_meta

    file_type = os.path.splitext(file_path)[1]

    if file_type == ".dpx":
        with open(file_path, 'wb') as f:
            dpx.writeDPX(f, img, last_meta if meta == None else meta)
    else:
        img = (img * 255.0).astype(np.uint8)
        misc.imsave(file_path, img)

# test code

def test_list(folder_path,deep=False):

    files = image_files(folder_path,deep)

    print("Listing " + folder_path)

    for image_type in files:
        if len(image_type) > 0:
            fname = image_type[len(image_type)//4]
            fname_info = os.path.splitext(os.path.basename(fname))
            print("Type  : " + fname_info[1])
            print("Number: " + str(len(image_type)))
            img = imread(fname)
            shape = np.shape(img)
            print("File  : " + fname)
            print("Shape : " + str(shape))
            print("Center: " + str(img[shape[0]//2][shape[1]//2]))
            print("")
            tfilename = 'X-'+''.join(fname_info)
            imsave(tfilename,img)
            img2 = imread(tfilename)
            print('save/load OK!' if np.array_equal(img,img2) else 'save/load bad!')
