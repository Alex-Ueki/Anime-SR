"""
frameops.py

Read, write and manipulate frame image files.

DPX code from https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc (Jack Doerner)

"""

import numpy as np
import scipy.misc as misc
import os
import copy
import random
import psutil
from skimage import transform as tf

import Modules.dpx as dpx

IMAGETYPES = ['.jpg', '.png', '.dpx']

# Simple in-memory cache of decoded image tiles. Since we are constantly cycling
# through the same images, we use a FINO (First-In-Never-Out) cache.

cached_tiles = {}
cached_quality = {}
caching = True
MINFREEMEMORY = 300 * 1000 * 1000

# In case we ever need to reset or disable the cache


def reset_cache(enabled=True):

    global cached_images
    global cached_quality
    global caching

    cached_images = {}
    cached_quality = {}
    caching = enabled

# Look in folder_path for all the files that are of one of the IMAGETYPES,
# and return a sorted list of lists containing the absolute paths to those files. So if
# there are only png files, the result will be a list containing a single list, but if
# there are jpg and dpx files, there will be two sublists, one for each type.
#
# If deep is True, look at all the subdirectories as well.


def image_files(folder_path, deep=False):

    file_paths = []
    file_lists = []

    for (root, dirs, files) in os.walk(folder_path):
        file_paths.extend([os.path.join(root, f) for f in files])
        if not deep:
            break

    if len(file_paths) == 0:
        return file_lists

    for ext in IMAGETYPES:
        ext_list = [f for f in file_paths if os.path.splitext(f)[1] == ext]
        if len(ext_list) > 0:
            file_lists.append(sorted(ext_list))

    return file_lists

# Clears all image files in a folder. Used in imgdistribute


def clear_image_files(folder_path, deep=False):

    file_paths = []
    file_lists = []

    for (root, dirs, files) in os.walk(folder_path):
        file_paths.extend([os.path.join(root, f) for f in files])
        if not deep:
            break

    if len(file_paths) == 0:
        return

    for ext in IMAGETYPES:
        ext_list = [f for f in file_paths if os.path.splitext(f)[1] == ext]
        for ext_file in ext_list:
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    assert len(image_files(folder_path, deep)) == 0, "clear_image_files failed"

# Read an image file and return numpy array of pixel values. Extends scipy.misc.imread
# with support for 10-bit DPX files. Always returns RGB images, with pixel values
# normalized to 0..1 range (inclusive).
#
# Note that the dpx library routines already handle (de-)normalization, so we only
# have to handle that for operations we hand off to scipy.misc.


def imread(file_path):

    # global last_meta

    file_type = os.path.splitext(file_path)[1]

    if os.path.isfile(file_path):
        if file_type == '.dpx':
            with open(file_path, 'rb') as f:
                img = dpx.DPXread(f)
        else:
            img = misc.imread(file_path, mode='RGB')
            img = img.astype('float32') / 255.0
    else:
        img = None

    return img

# Write a numpy array to an image file. Extends scipy.misc.imsave
# with support for 10-bit DPX files. Expects the input array to be
# normalized to 0..1 range (inclusive). If DPX is being saved, then
# if meta information is None, the meta information of the Last
# image loaded will be used.


def imsave(file_path, img, meta=None):

    # global last_meta

    file_type = os.path.splitext(file_path)[1]

    if file_type == '.dpx':
        with open(file_path, 'wb') as f:
            dpx.DPXsave(f, img)
    else:
        img = (img * 255.0).astype(np.uint8)
        misc.imsave(file_path, img)

# Generator for image tiles. Trims each image file (useful for handling 4:3 images in 16:9 HD files),
# then adds a border before generating the tiles. Each tile will be of shape
# (tile_height+border*2, tile_width+border*2, 3)
#
#   file_paths      list of image files to tesselate
#   tile_width      width of tile in image files
#   tile_height     height of tile in image files
#   border          number of pixels to add to tile borders
#   black_level     black color to use when adding borders
#   border_mode     border fill mode ('constant' or 'edge')
#   trim_...        pixels to trim from images before tesselating
#   shuffle         Do a random permutation of files
#   jitter          Jitter the tile position randomly
#   skip            Randomly skip 0 to 4 tiles between returned tiles
#   expected_       Expected width and height, default is HD
#   quality         Quality threshold for tiles. The fraction of tiles in
#                   an image that are returned, sorted by amount of detail
#                   in the tile. Default = 1.0 (use all tiles)
#   theano          If true, convert to theano data ordering
#
# Warning: reset the cache between uses of tesselate() and tesselate_pair(), or if quality changes!


def tesselate(file_paths, tile_width, tile_height, border, black_level=0.0, border_mode='constant',
              trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
              shuffle=True, jitter=False, skip=False,
              expected_width=1920, expected_height=1080,
              quality=1.0, theano=False):

    # Convert non-list to list

    file_paths = file_paths if type(file_paths) in (
        list, tuple) else [file_paths]

    # Shuffle the list

    if shuffle:
        random.shuffle(file_paths)

    # Process image files

    for f in file_paths:

        # Extract tiles from image

        if f in cached_tiles:
            tiles = cached_tiles[f]
        else:
            tiles = extract_tiles(f, tile_width, tile_height, border, black_level, border_mode,
                                  trim_top, trim_bottom, trim_left, trim_right, jitter,
                                  expected_width, expected_height, quality, theano=theano)

        if tiles != []:

            # If we are doing quality selection, then we need to fix the cache the first time
            # through.

            if quality < 1.0 and f not in cached_quality:
                tiles, _ = update_cache_quality(f, tiles)

            # Shuffle tiles

            tiles_list = list(range(len(tiles)))
            if shuffle:
                random.shuffle(tiles_list)

            # Generate tiles

            skip_count = random.randint(0, 4) if skip else 0

            for tile_index in tiles_list:
                if skip_count <= 0:
                    skip_count = random.randint(0, 4) if skip else 0
                    yield tiles[tile_index]
                else:
                    skip_count -= 1

# This version tesellates matched pairs of images, with identical shuffling behavior. Used for model training


def tesselate_pair(alpha_paths, beta_paths, tile_width, tile_height, border, black_level=0.0, border_mode='constant',
                   trim_top=0, trim_bottom=0, trim_left=0, trim_right=0, shuffle=True, jitter=False, skip=False,
                   expected_width=1920, expected_height=1080, quality=1.0, theano=False, residual=False):

    # Convert non-lists to lists
    alpha_paths = alpha_paths if type(alpha_paths) in (
        list, tuple) else [alpha_paths]
    beta_paths = beta_paths if type(beta_paths) in (
        list, tuple) else [beta_paths]

    all_paths = list(zip(alpha_paths, beta_paths))

    # Shuffle the lists

    if shuffle:
        random.shuffle(all_paths)

    # Process the image file pairs

    for alpha_path, beta_path in all_paths:

        # Extract the tiles from paired image files. One gotcha to keep in mind - there will be a
        # great disturbance in the force if the alpha tiles get cached but the beta tiles don't,
        # AND we happen to be messing with the tiles because of quality < 1.0. So we make sure
        # that only the read of the beta tiles can turn off the cache (which happens after a
        # cache store)

        if alpha_path in cached_tiles:
            alpha_tiles = cached_tiles[alpha_path]
        else:
            alpha_tiles = extract_tiles(alpha_path, tile_width, tile_height, border, black_level, border_mode,
                                        trim_top, trim_bottom, trim_left, trim_right, jitter,
                                        expected_width, expected_height, must_cache=caching, theano=theano)

        if beta_path in cached_tiles:
            beta_tiles = cached_tiles[beta_path]
        else:
            beta_tiles = extract_tiles(beta_path, tile_width, tile_height, border, black_level, border_mode,
                                       trim_top, trim_bottom, trim_left, trim_right, jitter,
                                       expected_width, expected_height, must_cache=True, theano=theano)

        if len(alpha_tiles) != len(beta_tiles):
            print('Tesselation error: file pairs {} and {} have different numbers of tiles {} and {}'.format(
                alpha_path, beta_path, len(alpha_tiles), len(beta_tiles)))
        elif len(alpha_tiles) > 0:
            # If we are doing quality selection, then we need to fix the cache the first time
            # through. The trick, however, is that we need to use the quality ratings for
            # the beta tiles to determine the order of the alpha tiles, so things remain
            # properly paired.

            if quality < 1.0 and beta_path not in cached_quality:
                beta_tiles, beta_indexes = update_cache_quality(
                    beta_path, beta_tiles, quality)
                alpha_tiles, _ = update_cache_quality(
                    alpha_path, alpha_tiles, quality, beta_indexes)

            # Shuffle tiles

            tiles_list = list(range(len(alpha_tiles)))
            if shuffle:
                random.shuffle(tiles_list)

            # Generate tiles

            skip_count = random.randint(0, 4) if skip else 0

            for tile_index in tiles_list:
                if skip_count <= 0:
                    skip_count = random.randint(0, 4) if skip else 0
                    # GPU : PU please check this
                    if residual: # returns the difference between beta-alpha
                        yield (alpha_tiles[tile_index], beta_tiles[tile_index] - alpha_tiles[tile_index])
                    else:
                        yield (alpha_tiles[tile_index], beta_tiles[tile_index])
                else:
                    skip_count -= 1

# Handle adjusting the tile cache when we are doing quality determinations. Returns the
# new tiles and the sort indexes. Our quality value is the negated sum of the pixel
# value differences of adjacent pixels.


def update_cache_quality(path, tiles, quality, indexes=None):

    global cached_tiles
    global cached_quality

    # Compute the sorting indexes.

    if indexes == None:
        indexes = [(idx, -np.sum(np.absolute(tile[:, 1:, :] - tile[:, :-1, :])))
                   for idx, tile in enumerate(tiles)]
        indexes.sort(key=lambda x: x[1])
        indexes = indexes[:max(1, int(len(indexes) * quality))]
        indexes = [index[0] for index in indexes]

    # Reshuffle the tiles (also reduces them to the correct number)

    tiles = [tiles[i] for i in indexes]

    # Update the cache if the tiles are in there

    if path in cached_tiles:
        cached_tiles[path] = tiles
        cached_quality[path] = True

    return (tiles, indexes)

# Helper function that reads in a file, extracts the tiles, and caches them if possible. Handles
# size conversion if needed.


resize_warning = True


def extract_tiles(file_path, tile_width, tile_height, border, black_level=0.0, border_mode='constant',
                  trim_top=0, trim_bottom=0, trim_left=0, trim_right=0, jitter=False,
                  expected_width=1920, expected_height=1080, must_cache=True, theano=False):

    # global last_meta
    global cached_tiles
    global caching
    global resize_warning

    img = imread(file_path)

    # If we just read in an image that is not the expected size, we need to scale.
    # The resolutions we currently are likely to see are 640x480, 720x480 and
    # 720x486. In the latter case we chop off 3 rows top and bottom to get 720x480
    # before scaling. When we upscale, we do so into the trimmed image area.

    shape = np.shape(img)

    if shape[0] > expected_height or shape[1] > expected_width:
        if resize_warning:
            print(
                'Warning: Read image larger than expected {} - downscaling'.format(shape))
            print('(This warning will not repeat)')
            resize_warning = False
        img = tf.resize(img,
                        (expected_height, expected_width, 3),
                        order=1,
                        mode='constant')
    elif shape[0] < expected_height or shape[1] < expected_width:

        if resize_warning:
            print(
                'Warning - Read image smaller than expected {} - upscaling'.format(shape))

        # Handle 486 special-case

        if shape[0] == 486:
            img = img[3:-3, :, :]
            shape = np.shape(img)
            if resize_warning:
                print('Scaled from 486v to {}'.format(shape))

        trimmed_height = expected_height - trim_top - trim_bottom
        trimmed_width = expected_width - trim_left - trim_right

        img = tf.resize(img,
                        (trimmed_height, trimmed_width, 3),
                        order=1,
                        mode='constant')

        if resize_warning:
            print('Uprezzed to {}'.format(np.shape(img)))
            print('(This warning will not repeat)')
            resize_warning = False

    elif trim_top + trim_bottom + trim_left + trim_right > 0:

        # Input image is expected size, but we have to trim it

        img = img[trim_top:shape[0] - trim_bottom,
                  trim_left:shape[1] - trim_right, :]

    # Shape may have changed due to all of the munging above

    shape = np.shape(img)

    # Generate image tile offsets

    shape = np.shape(img)
    rows = shape[0] // tile_height
    cols = shape[1] // tile_width

    if len(shape) != 3 or (shape[0] > rows * tile_height) or (shape[1] > cols * tile_width):
        print('Tesselation Error: file {} has incorrect shape {}'.format(
            file_path, str(shape)))
        return []

    # Pad the image - if border_mode is 'constant', the pixels added have
    # value (black_level, black_level, black_level). If the mode is 'edge',
    # they copy the edge values.

    if border_mode == 'constant':
        img = np.pad(img,
                     ((border, border), (border, border), (0, 0)),
                     mode=border_mode, constant_values=black_level)
    else:
        img = np.pad(img,
                     ((border, border), (border, border), (0, 0)),
                     mode=border_mode)

    # Actual tile width and height

    across, down = tile_width + (2 * border), tile_height + (2 * border)

    # Unjittered tile offsets

    offsets = [(row * tile_height, col * tile_width)
               for row in range(0, rows) for col in range(0, cols)]

    # Jittered offsets are shifted half a tile across and down

    if jitter:
        half_across = across // 2
        half_down = down // 2
        jittered_offsets = [(row * tile_height + half_down, col * tile_width + half_across)
                            for row in range(0, rows - 1) for col in range(0, cols - 1)]
        offsets.extend(jittered_offsets)

    # Extract tiles from the image

    tiles = [img[rpos:rpos + down, cpos:cpos + across, :]
             for (rpos, cpos) in offsets]

    # Theano transposition (I hope!)

    if theano:
        tiles = tiles.transpose((0, 3, 1, 2))

    # Cache the tiles if possible. We can make sure the cache doesn't turn off by
    # setting must_cache. This lets us ensure that pairs of tiles are both cached.

    if caching:
        cached_tiles[file_path] = tiles
        mem = psutil.virtual_memory()
        if caching and (not must_cache) and mem.free < MINFREEMEMORY:
            caching = False
            print('')
            print('-----------------------------------------')
            print('Cache is full : {} images in cache'.format(len(cached_tiles)))
            print('Memory status : {}'.format(mem))
            print('MINFREEMEMORY : {}'.format(MINFREEMEMORY))
            print('-----------------------------------------')
            print('')

    return tiles

# Quasi-inverse to tesselate; glue together tiles to form a final image; takes list of numpy tile arrays,
# trims off the borders, stitches them together, and pads as needed.
#
#   tiles           1-d list of numpy image tiles
#   border          number of pixels to remove from the border
#   row_width       number of tiles per row (number of rows is thus implicit)
#   black_level    black color to use when padding
#   pad_...         amount of padding to create on each edge
#   theano          if True, transpose the array into theano order. UNTESTED


def grout(tiles, border, row_width, black_level=0.0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, theano=False):

    # Figure out the size of the final image and allocate it

    tile_shape = np.shape(tiles[0])
    tile_width = tile_shape[1] - border * 2
    tile_height = tile_shape[0] - border * 2

    row_count = (len(tiles) // row_width)
    img_width, img_height = tile_width * row_width, tile_height * row_count

    img = np.empty((img_height, img_width, 3), dtype='float32')

    # Tile clipping range

    first_col, last_col = border, tile_shape[1] - border
    first_row, last_row = border, tile_shape[0] - border

    # Theano transposition (I hope!)

    if theano:
        tiles = tiles.transpose((0, 3, 1, 2))

    # Grout the tiles

    cur_tile = 0
    for row in range(row_count):
        img_row = row * tile_height
        for col in range(row_width):
            img_col = col * tile_width
            img[img_row:img_row + tile_height, img_col:img_col +
                tile_width] = tiles[cur_tile][first_row:last_row, first_col:last_col]
            cur_tile += 1

    # Pad the tiles

    if pad_top + pad_bottom + pad_left + pad_right > 0:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right),
                           (0, 0)), mode='constant', constant_values=black_level)

    return img

# Compute the size of the tile grid for an image


def tile_grid(file_path, tile_width=60, tile_height=60, border=2, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0):

    # We might get passed a list, unpack until we get a string (filepath)

    while type(file_path) is list:
        file_path = file_path[0]

    img = imread(file_path)
    shape = np.shape(img)

    rows = shape[0] - (trim_top + trim_bottom) // tile_height
    cols = shape[1] - (trim_left + trim_right) // tile_width

    return (rows, cols)

# Turn a tile generator into a tile batch generator
# Currently not used, was having a problem getting the model to
# predict with a generator. See model.predict_tiles; it was
# expecting more tiles than it actually got.


def batch_generator(tile_generator, image_shape, batch_size):

    tiles = np.empty((batch_size, ) + image_shape)
    batch_index, bnum = 0, 1
    print('batch size ', batch_size)

    # Generate batches of tiles

    for tile in tile_generator:

        # PU: This is already handled inside modelio.py

        # if K.image_dim_ordering() == 'th':
        #    tile = tile.transpose((2, 0, 1))

        tiles[batch_index] = tile
        batch_index += 1
        if batch_index >= batch_size:
            batch_index = 0
            print('Yielding tile batch ', bnum)
            bnum += 1
            yield tiles

    # Output residual tiles

    if batch_index > 0:
        tiles = tiles[:batch_index]
        print('Yielding residual tiles')
        yield tiles
