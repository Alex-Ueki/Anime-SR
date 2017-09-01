# Anime Image Super Resolution using in Keras 2+ (TensorFlow Backend)

Credit to
*<a href="https://github.com/titu1994/Image-Super-Resolution">Image-Super-Resolution</a>*
for base code, https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc for DPX file io.

## Important note for GPU

Look for "# PU" comments in models.py for questions about "PeekSignaltoNoiseRatio"

## Setup

Set up Data directory as follows:

```
Data
    eval_images/    
      Alpha/    
      Beta/   
    input_images/   
      Alpha/    
      Beta/   
    predict_images/   
      Alpha/    
    train_images/           Images to use for training
      training/                 Actual training images
        Alpha/                      Input images
        Beta/                       Target images
      validation/               Model validation images
        Alpha/                      Input images
        Beta/                       Target images
```

## Usage

```
train.py [model] [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR
        All

    All: train all models sequentially with the same options

Options are:

    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          epoch size, default=255
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240
    trimright=nnn       pixels to trim on image right edge, default = 240
    trimtop=nnn         pixels to trim on image top edge, default = 0
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to model file, default = Weights/{model}-{width}-{height}-{border}.h5
    history=path        path to checkpoint file, default = Weights/{model}-{width}-{height}-{border}_history.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)
```
## TODOS

- Parental Unit
    - Add skip tile feature to trainer?
    - Some way of skipping low-contrast tiles? Use blz to compress tiles and check size? Probably too slow though.
    - Better error handling on file IO
    - Create path structure in Data folder if necessary
    - Path structure validation improvements
    - Update dpx.py to extract black levels automatically.
    - Write evaluate.py, hallucinate.py, etc.
    - Cleanup pass to conform to PEP 8 (find pretty-printer for python/atom)
    - Improve Documentation
- Gene-Perpetuation Unit
    - New, better models
    - Random Brilliance

## Model Architecture (From Image-Super-Resolution)

The following are the models sourced from Image-Super-Resolution

1. Super Resolution CNN (SR)
2. Expanded Super Resolution CNN (ESR)
3. Deep Denoiseing Super Resolution (DDSR)

There is also an incomplete implementation of *<a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a>*.

1. Very Deep Super Resolution (VDSR)

Note that all models are currently designed to work with 64x64 images.
