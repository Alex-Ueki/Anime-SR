# pylint: disable=C0301, W0603, R0913, R0914
# (line_too_long, used_globals, too_many_variables, too_many_arguments)

""" frameops.py
    Read, write and manipulate frame image files. Create generators for Keras.
    DPX code from https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc (Jack Doerner)
"""

import os
import random
import psutil

import numpy as np
import scipy.misc as misc
from skimage import transform as tf

import Modules.dpx as dpx
from Modules.misc import printlog

#from Modules.modelio import ModelIO

IMAGETYPES = ['.jpg', '.png', '.dpx']

# Simple in-memory cache of decoded image tiles. Since we are constantly cycling
# through the same images, we use a FINO (First-In-Never-Out) cache. These are
# mutable globals!

CACHED_TILES = {}
CACHED_QUALITY = {}
CACHING = True
RESIZE_WARNING = True

# Minimum amount of free memory permitted (after which we stop adding to the
# cache)

MINFREEMEMORY = 300 * 1000 * 1000

# In case we ever need to reset or disable the cache


def reset_cache(enabled=True):
    """ Reset and enable/disable the tile cache """

    global CACHED_TILES
    global CACHED_QUALITY
    global CACHING

    CACHED_TILES = {}
    CACHED_QUALITY = {}
    CACHING = enabled


def image_files(folder_path, deep=False):
    """ Look in folder_path for all the files that are of one of the IMAGETYPES,
        and return a sorted list of lists containing the absolute paths to those files. So if
        there are only png files, the result will be a list containing a single list, but if
        there are jpg and dpx files, there will be two sublists, one for each type.

        If deep is True, look at all the subdirectories as well.
    """

    file_paths = []
    file_lists = []

    for (root, _, files) in os.walk(folder_path):
        file_paths.extend([os.path.join(root, f) for f in files])
        if not deep:
            break

    if not file_paths:
        return file_lists

    for ext in IMAGETYPES:
        ext_list = [f for f in file_paths if os.path.splitext(f)[1] == ext]
        if ext_list:
            file_lists.append(sorted(ext_list))

    return file_lists


def imread(file_path):
    """ Read an image file and return numpy array of pixel values. Extends scipy.misc.imread
        with support for 10-bit DPX files. Always returns RGB images, with pixel values
        normalized to 0..1 range (inclusive).

        Note that the dpx library routines already handle (de-)normalization, so we only
        have to handle that for operations we hand off to scipy.misc.
    """

    # global last_meta

    file_type = os.path.splitext(file_path)[1]

    if os.path.isfile(file_path):
        if file_type == '.dpx':
            with open(file_path, 'rb') as dpxfile:
                img = dpx.read(dpxfile)
        else:
            img = misc.imread(file_path, mode='RGB')
            img = img.astype('float32') / 255.0
    else:
        img = None

    return img


def imsave(file_path, img, meta=None):
    """ Write a numpy array to an image file. Extends scipy.misc.imsave
        with support for 10-bit DPX files. Expects the input array to be
        normalized to 0..1 range (inclusive). If DPX is being saved, then
        if meta information is None, the meta information of the Last
        image loaded will be used.
    """

    # global last_meta

    file_type = os.path.splitext(file_path)[1]

    if file_type == '.dpx':
        with open(file_path, 'wb') as dpxfile:
            dpx.save(dpxfile, img, meta)
    else:
        img = (img * 255.0).astype(np.uint8)
        misc.imsave(file_path, img)


def tesselate(file_paths, config):
    """, tile_width, tile_height, border, black_level=0.0, border_mode='constant',
              trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
              shuffle=True, jitter=False, skip=False,
              expected_width=1920, expected_height=1080,
              quality=1.0, theano=False):
    """
    """ Generator for image tiles. Trims each image file (useful for handling 4:3 images in 16:9 HD files),
        then adds a border before generating the tiles. Each tile will be of shape
        (tile_height+border*2, tile_width+border*2, 3). Config is a ModelIO containing these values:

        tile_width      width of tile in image files
        tile_height     height of tile in image files
        border          number of pixels to add to tile borders
        black_level     black color to use when adding borders
        border_mode     border fill mode ('constant' or 'edge')
        trim_...        pixels to trim from images before tesselating
        shuffle         Do a random permutation of files
        jitter          Jitter the tile position randomly
        skip            Randomly skip 0 to 4 tiles between returned tiles
        expected_...    Expected width and height, default is HD
        quality         Quality threshold for tiles. The fraction of tiles in
                        an image that are returned, sorted by amount of detail
                        in the tile. Default = 1.0 (use all tiles)
        theano          If true, convert to theano data ordering

        Warning: reset the cache between uses of tesselate() and tesselate_pair(), or if quality changes!
    """

    # Convert non-list to list

    file_paths = file_paths if isinstance(file_paths, (list, tuple)) else [file_paths]

    # Shuffle the list

    if config.shuffle:
        random.shuffle(file_paths)

    # Process image files

    for img_file in file_paths:

        # Extract tiles from image

        if img_file in CACHED_TILES:
            tiles = CACHED_TILES[img_file]
        else:
            tiles = extract_tiles(img_file, config)

        if tiles != []:

            # If we are doing quality selection, then we need to fix the cache the first time
            # through.

            if config.quality < 1.0 and img_file not in CACHED_QUALITY:
                tiles, _ = update_cache_quality(img_file, tiles, config.quality)

            # Shuffle tiles

            tiles_list = list(range(len(tiles)))
            if config.shuffle:
                random.shuffle(tiles_list)

            # Generate tiles

            skip_count = random.randint(0, 4) if config.skip else 0

            for tile_index in tiles_list:
                if skip_count <= 0:
                    skip_count = random.randint(0, 4) if config.skip else 0
                    yield tiles[tile_index]
                else:
                    skip_count -= 1

#


def tesselate_pair(alpha_paths, beta_paths, config):
    """ This version tesellates matched pairs of images, with identical shuffling behavior. Used for model training """

    # Convert non-lists to lists

    alpha_paths = alpha_paths if isinstance(
        alpha_paths, (list, tuple)) else [alpha_paths]
    beta_paths = beta_paths if isinstance(
        beta_paths, (list, tuple)) else [beta_paths]

    all_paths = list(zip(alpha_paths, beta_paths))

    # Shuffle the lists

    if config.shuffle:
        random.shuffle(all_paths)

    # Process the image file pairs

    for alpha_path, beta_path in all_paths:

        # Extract the tiles from paired image files. One gotcha to keep in mind - there will be a
        # great disturbance in the force if the alpha tiles get cached but the beta tiles don't,
        # AND we happen to be messing with the tiles because of quality < 1.0. So we make sure
        # that only the read of the beta tiles can turn off the cache (which happens after a
        # cache store)

        alpha_tiles = extract_tiles(alpha_path, config, can_disable=False)

        beta_tiles = extract_tiles(beta_path, config, can_disable=True)

        if len(alpha_tiles) != len(beta_tiles):
            printlog('Tesselation error: file pairs {} and {} have different tile counts {} and {}'.format(
                alpha_path, beta_path, len(alpha_tiles), len(beta_tiles)))
        elif alpha_tiles:
            # If we are doing quality selection, then we need to fix the cache the first time
            # through. The trick, however, is that we need to use the quality ratings for
            # the beta tiles to determine the order of the alpha tiles, so things remain
            # properly paired.

            if config.quality < 1.0 and beta_path not in CACHED_QUALITY:
                beta_tiles, beta_indexes = update_cache_quality(
                    beta_path, beta_tiles, config.quality)
                alpha_tiles, _ = update_cache_quality(
                    alpha_path, alpha_tiles, config.quality, beta_indexes)

            # Shuffle tiles

            tiles_list = list(range(len(alpha_tiles)))
            if config.shuffle:
                random.shuffle(tiles_list)

            # Generate tiles

            skip_count = random.randint(0, 4) if config.skip else 0

            for tile_index in tiles_list:
                if skip_count <= 0:
                    skip_count = random.randint(0, 4) if config.skip else 0
                    # GPU : PU please check this
                    if config.residual: # returns the difference between beta-alpha
                        yield (alpha_tiles[tile_index], beta_tiles[tile_index] - alpha_tiles[tile_index])
                    else:
                        yield (alpha_tiles[tile_index], beta_tiles[tile_index])
                else:
                    skip_count -= 1


def update_cache_quality(path, tiles, quality, indexes=None):
    """ Handle adjusting the tile cache when we are doing quality determinations. Returns the
        new tiles and the sort indexes. Our quality value is the negated sum of the pixel
        value differences of adjacent pixels. To match one set of tiles to another, pass
        back the sort indexes from the first call.
    """

    global CACHED_TILES
    global CACHED_QUALITY

    # Compute the sorting indexes if we are not given them already.

    if not indexes:
        indexes = [(idx, -1 * np.sum(np.absolute(tile[:, 1:, :] - tile[:, :-1, :])))
                   for idx, tile in enumerate(tiles)]
        indexes.sort(key=lambda x: x[1])
        indexes = indexes[:max(1, int(len(indexes) * quality))]
        indexes = [index[0] for index in indexes]

    # Reshuffle the tiles (also reduces them to the correct number)

    tiles = [tiles[i] for i in indexes]

    # Update the cache if the tiles are in there

    if path in CACHED_TILES:
        CACHED_TILES[path] = tiles
        CACHED_QUALITY[path] = True

    return (tiles, indexes)


def extract_tiles(file_path, config, can_disable=False):
    """ Helper function that reads in a file, extracts the tiles, and caches them if possible. Handles
        size conversion if needed. Note that it cannot handle the quality tile reduction since that
        has to be matched between the alpha and beta tiles
    """

    global CACHED_TILES
    global CACHING
    global RESIZE_WARNING

    # Cache hit?

    if file_path in CACHED_TILES:
        return CACHED_TILES[file_path]

    img = imread(file_path)

    # If we just read in an image that is not the expected size, we need to scale.
    # The resolutions we currently are likely to see are 640x480, 720x480 and
    # 720x486. In the latter case we chop off 3 rows top and bottom to get 720x480
    # before scaling. When we upscale, we do so into the trimmed image area.

    shape = np.shape(img)

    if shape[0] > config.image_height or shape[1] > config.image_width:
        if RESIZE_WARNING:
            printlog('Warning: Read image larger than expected {} - downscaling'.format(shape))
            printlog('(This warning will not repeat)')
            RESIZE_WARNING = False
        img = tf.resize(img,
                        (config.image_height, config.image_width, 3),
                        order=1,
                        mode='constant')
    elif shape[0] < config.image_height or shape[1] < config.image_width:

        # Handle 486 special-case

        if shape[0] == 486:
            img = img[3:-3, :, :]
            shape = np.shape(img)

        #trimmed_height = config.expected_height - config.trim_top - config.trim_bottom
        #trimmed_width = expected_width - trim_left - trim_right

        img = tf.resize(img,
                        (config.trimmed_height, config.trimmed_width, 3),
                        order=1,
                        mode='constant')

        if RESIZE_WARNING:
            printlog('Warning - Read image smaller than expected {} - upscaled to {}'.format(shape, np.shape(img)))
            printlog('(This warning will not repeat)')
            RESIZE_WARNING = False

    elif config.trim_top + config.trim_bottom + config.trim_left + config.trim_right > 0:

        # Input image is expected size, but we have to trim it

        img = img[config.trim_top:shape[0] - config.trim_bottom,
                  config.trim_left:shape[1] - config.trim_right, :]

    # Shape may have changed due to all of the munging above

    shape = np.shape(img)

    # Generate image tile offsets

    if len(shape) != 3 or (shape[0] > config.tiles_down * config.base_tile_height) or (shape[1] > config.base_tile_width * config.tiles_across):
        printlog('Tesselation Error: file {} has incorrect shape {}'.format(file_path, str(shape)))
        return []

    # Pad the image - if border_mode is 'constant', the pixels added have
    # value (black_level, black_level, black_level). If the mode is 'edge',
    # they copy the edge values.

    if config.border_mode == 'constant':
        img = np.pad(img,
                     ((config.border, config.border), (config.border, config.border), (0, 0)),
                     mode=config.border_mode, constant_values=config.black_level)
    else:
        img = np.pad(img,
                     ((config.border, config.border), (config.border, config.border), (0, 0)),
                     mode=config.border_mode)

    # Actual tile width and height

    #across, down = configtile_width + (2 * border), tile_height + (2 * border)

    # Unjittered tile offsets

    offsets = [(row * config.base_tile_height, col * config.base_tile_width)
               for row in range(0, config.tiles_down) for col in range(0, config.tiles_across)]

    # Jittered offsets are shifted half a tile across and down

    if config.jitter:
        half_across = config.tile_width // 2
        half_down = config.tile_height // 2
        jittered_offsets = [(row * config.base_tile_height + half_down, col * config.base_tile_width + half_across)
                            for row in range(0, config.tiles_down - 1) for col in range(0, config.tiles_across - 1)]
        offsets.extend(jittered_offsets)

    # Extract tiles from the image

    tiles = [img[rpos:rpos + config.tile_height, cpos:cpos + config.tile_width, :]
             for (rpos, cpos) in offsets]

    # Theano transposition (I hope!)

    if config.theano:
        tiles = tiles.transpose((0, 3, 1, 2))

    # Cache the tiles if possible. We can make sure the cache doesn't turn off by
    # setting must_cache. This lets us ensure that pairs of tiles are both cached.

    if CACHING:
        CACHED_TILES[file_path] = tiles
        mem = psutil.virtual_memory()
        if can_disable and mem.free < MINFREEMEMORY:
            CACHING = False
            print('')
            print('-----------------------------------------')
            print('Cache is full : {} images in cache'.format(len(CACHED_TILES)))
            print('Memory status : {}'.format(mem))
            print('MINFREEMEMORY : {}'.format(MINFREEMEMORY))
            print('-----------------------------------------')
            print('')

    return tiles


def grout(tiles, config):
    """ border, row_width, black_level=0.0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, theano=False): """
    """ Quasi-inverse to tesselate; glue together tiles to form a final image; takes list of numpy tile arrays,
        trims off the borders, stitches them together, and pads as needed.

        tiles           1-d list of numpy image tiles
        border          number of pixels to remove from the border
        row_width       number of tiles per row (number of rows is thus implicit)
        black_level    black color to use when padding
        pad_...         amount of padding to create on each edge
        theano          if True, transpose the array into theano order. UNTESTED
    """

    # Figure out the size of the final image and allocate it

    #tile_shape = np.shape(tiles[0])
    #tile_width = tile_shape[1] - border * 2
    #tile_height = tile_shape[0] - border * 2

    #row_count = (len(tiles) // row_width)
    #img_width, img_height = tile_width * row_width, tile_height * row_count

    img = np.empty((config.trimmed_height, config.trimmed_width, 3), dtype='float32')

    # Tile clipping range

    first_col, last_col = config.border, config.tile_width - config.border
    first_row, last_row = config.border, config.tile_height - config.border

    # Theano transposition (I hope!)

    if config.theano:
        tiles = tiles.transpose((0, 3, 1, 2))

    # Grout the tiles

    cur_tile = 0
    for row in range(config.tiles_down):
        img_row = row * config.base_tile_height
        for col in range(config.tiles_across):
            img_col = col * config.base_tile_width
            img[img_row:img_row + config.base_tile_height, img_col:img_col +
                config.base_tile_width] = tiles[cur_tile][first_row:last_row, first_col:last_col]
            cur_tile += 1

    # Pad the tiles

    if config.trim_top + config.trim_bottom + config.trim_left + config.trim_right > 0:
        img = np.pad(img, ((config.trim_top, config.trim_bottom), (config.trim_left, config.trim_right),
                           (0, 0)), mode='constant', constant_values=config.black_level)

    return img


def batch_generator(tile_generator, image_shape, batch_size):
    """ Turn a tile generator into a tile batch generator
        Currently not used, was having a problem getting the model to
        predict with a generator. See model.predict_tiles; it was
        expecting more tiles than it actually got.
    """

    tiles = np.empty((batch_size, ) + image_shape)
    batch_index, bnum = 0, 1

    # Generate batches of tiles

    for tile in tile_generator:

        # PU: This is already handled inside modelio.py

        # if K.image_dim_ordering() == 'th':
        #    tile = tile.transpose((2, 0, 1))

        tiles[batch_index] = tile
        batch_index += 1
        if batch_index >= batch_size:
            batch_index = 0
            bnum += 1
            yield tiles

    # Output residual tiles

    if batch_index > 0:
        tiles = tiles[:batch_index]
        yield tiles
