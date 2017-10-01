# pylint: disable=C0301
# (line_too_long disabled)
"""
python3 dpxderez.py {source directory} {destination directory}

Converts 1920x1080 HD DPX down to 720x480

"""

import sys
import os
from skimage import transform as tf

import Modules.dpx as dpx

def derez():
    """ Downconvert dpx images """

    fromdir = sys.argv[1]
    todir = sys.argv[2]

    fnames = []
    for (_, _, filenames) in os.walk(fromdir):
        for fname in filenames:
            if fname.endswith('.dpx'):
                fnames.append(fname)
        break

    if not fnames:
        print('No dpx files found in', fromdir)
        exit(1)

    for filename in fnames:
        fpath = os.path.join(fromdir, filename)
        tpath = os.path.join(todir, filename)
        if not os.path.exists(tpath):
            print("Processing: " + fpath)
            dpxfile = open(fpath, "rb")
            image = dpx.read(dpxfile)
            image = image[:, 240:-240, :]
            downrez = tf.resize(image, (480, 720, 3), order=1, mode='constant')
            dpxfile = open(tpath, "wb")
            dpx.save(dpxfile, downrez)

if __name__ == '__main__':
    derez()
