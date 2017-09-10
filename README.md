# Anime Image Super Resolution using in Keras 2+ (TensorFlow Backend)

Credit to
*<a href="https://github.com/titu1994/Image-Super-Resolution">Image-Super-Resolution</a>*
for base code, *https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc* for DPX file io.

## Setup

Data directory should be as follows:

```
Data
    eval_images/    
        Alpha/    
        Beta/   
    input_images/   
        Alpha/    
        Beta/   
    predict_images/         Images to predict
        Alpha/                  Source images
        Beta/                   Predicted images
    train_images/           Images to use for training
        training/               Actual training images
            Alpha/                  Input images
            Beta/                   Target images
        validation/             Model validation images
            Alpha/                  Input images
            Beta/                   Target images
    models/                 .h5 and _state.json files for trained models.
```

Alex : *<a href="https://www.dropbox.com/sh/69ec5l61cpmsgmn/AAAIAdlsfDe6_hZorAfu3yIwa?dl=0&preview=Data.zip">Data.zip</a>*
## Usage

```
train.py [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    type=model          model type, default is BasicSR
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          epoch size, default=9999
    lr=.nnn             set initial learning rate, default = use model's current learning rate. Should be 0.001 or less.
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240
    trimright=nnn       pixels to trim on image right edge, default = 240
    trimtop=nnn         pixels to trim on image top edge, default = 0
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0
    jitter=1|0|T|F      include jittered tiles (offset by half a tile across&down) when training; default=True
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=True
    shuffle=1|0|T|F     shuffle tiles into random order when training; default=True
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to trained model file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}.h5
    state=path          path to state file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}_state.txt

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

predict.py [option(s)] ...

    Predicts images by applying model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/predict_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information

```

## Useful Tools

```
Tools/sdcopy.sh {src folder} {from folder} {dest folder}

    For each file in {src folder}, copy the file with the same name from {from folder}
    to {dest folder}. Handy for creating matched training / validation file sets. This
    is a bash shell script.

Tools/dpxderez.py {source folder} {destination folder}

    For each 1920x1080 DPX image file in {source folder}, assume it contains a 1440x1080
    4x3 image. Crop to this and crunch it down to 720x480 (SD resolution), then scale it
    back to 1440x1080 again, add black bars to make it 1920x1080, and save in the
    {destination folder}. Lets you create the lower-resolution inputs to the models.
```

## TODOS

- Parental Unit
    - Test current models
    - Pythonate sdcopy.sh
    - Improve Documentation

- Gene-Perpetuation Unit
    - Clean up contents of Scripts folder
    - Implement a function that takes images from input_images and divides them into train_images (for training and validation) and eval_images (for evaluation)
    - New, better models
    - Random Brilliance

## Model Architecture (From Image-Super-Resolution)

The following are the models sourced from Image-Super-Resolution

1. Super Resolution CNN (SR)
2. Expanded Super Resolution CNN (ESR)
3. Deep Denoising Super Resolution (DDSR)

There is also an incomplete implementation of *<a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a>*.

1. Very Deep Super Resolution (VDSR)

Note that all models are currently designed to work with 64x64 images.
