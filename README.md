# Anime Image Super Resolution using in Keras 2+ (TensorFlow Backend)

Toolkit for investigating techniques for improved upconversion of SD video to HD. Can potentially be adapted for
use on other image conversion tasks.

## Features

- Tiling of images with overlapping borders.
- Quality-culling of tiles based on approximate image detail.
- Tools for data preparation, training, evaluation, prediction, genetic evolution of models.
- Training can be interrupted and restarted freely.

Training dataset (remove this link before publishing repo): https://www.dropbox.com/sh/69ec5l61cpmsgmn/AAAIAdlsfDe6_hZorAfu3yIwa?dl=0

## Setup

Requires python 3.5+, Keras, assorted standard packages (numpy, scipy, etc.)

Data directory should be as follows (setup.py will do this for you)

```
Data
    eval_images/    
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

*<a href="https://www.imagemagick.org/script/index.php">ImageMagick</a>* is an indispensible tool for manipulating bitmap images.

See *Notes.md* for useful tips / tricks / comments

## Tool Usage


#### train.py [option(s)] ...
```
    Trains a model. See Modules/models.py for sample model types.

Options are:

    type=model          model type, default is BasicSR
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          max epoch count, default=100. See below for more details
    epochs+=nnn         run epoch count, default=None. Overrides epochs=nnn
    lr=.nnn             set initial learning rate, default = use model's current learning rate. Should be 0.001 or less
    quality=.nnn        fraction of the "best" tiles used in training (but not validation). Default is 1.0 (use all)
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240; can also use left=nnn
    trimright=nnn       pixels to trim on image right edge, default = 240; can also use right=nnn
    trimtop=nnn         pixels to trim on image top edge, default = 0; can also use top=nnn
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0; can also use bottom=nnn
    jitter=1|0|T|F      include tiles offset by half a tile across&down when training (but not validation); default=True
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=True
    shuffle=1|0|T|F     shuffle tiles into random order when training; default=True
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to trained model file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}.h5
    state=path          path to state file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}_state.json
    verbose=1|0|T|F     display verbose training output, default = True
    bargraph=1|0|T|F    display bargraph of training progress, default = True

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK).

    You can terminate a training session with ^C, and then resume training by reissuing the same command. The model's state
    is completely stored in the .h5 file, and the training state is in the _state.json file.

    The epochs value is the maximum number of epochs that will be trained **over multiple sessions**. So if you have
    previously trained a model for 50 epochs, epochs=75 would mean the model trains for 25 additional epochs. Alternately,
    you could specify epochs+=25 to limit the current training run to 25 epochs.
````
#### evaluate.py [option(s)] ...
````
    Evaluates quality of a model.

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/eval_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information
````
#### predict.py [option(s)] ...
````
    Predicts images using a model.

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/predict_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information
````
#### evolve.py [option(s)] ...
````
    Evolves (hopefully) improved models. In each generation, the 5 best models
    are retained, and then 15 new models are evolved from them. Progressively
    trains the new models one epoch at a time for 10 epochs, discarding the
    worst performer in each iteration.

Options are:

    genepool=path       path to genepool file, default is {Data}/genepool.json
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    lr=.nnn             set initial learning rate. Should be 0.001 or less. Default = 0.001
    quality=.nnn        fraction of the "best" tiles used in training (but not validation). Default is 1.0 (use all)
    residual=1|0|T|F    have the model train using residual images. Default=True.
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240; can also use left=nnn
    trimright=nnn       pixels to trim on image right edge, default = 240; can also use right=nnn
    trimtop=nnn         pixels to trim on image top edge, default = 0; can also use top=nnn
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0; can also use bottom=nnn
    jitter=1|0|T|F      include tiles offset by half a tile across&down when training (but not validation); default=True
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=True
    shuffle=1|0|T|F     shuffle tiles into random order when training; default=True
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK).

    Options are overridden by the contents of genepool.json, if any. Thus they are typically only specified on
    the first run. If t genepool.json file does not exist, it will be created with an initial population similar
    to some of the models in models.py

    See Modules/genomics.py for details on the structure of genes, codons and other genetic elements.
````
#### monitor.py
````
    Interactively displays the status of regular models (the ones in Data/models) and genetically evolved
    models (the data in Data/genepool.json). Has some options that make it easier to explore the statistics
    generated by evolve.py.
````
#### dpxderez.py {source folder} {destination folder}
````
    Expects {source folder} to contain 1920x1080 HD images with 1440x1080 4:3 framing. Converts to
    720x480 SD images and saves in {destination folder}. Used for data preparation.
````

## Miscellaneous Tools

#### Tools/sdcopy.sh {src folder} {from folder} {dest folder}
````
    For each file in {src folder}, copy the file with the same name from {from folder}
    to {dest folder}. Handy for creating matched training / validation file sets. This
    is a bash shell script.
````
#### Tools/setup.py
````
    Set up Data directory and subdirectories.
````
#### Tools/dpxinfo.py {dpx file path}
````
    Dumps DPX header info.
````

## TODOS

- Parental Unit
    - Test current models
    - Pythonate sdcopy.sh
    - Improve Documentation

- Gene-Perpetuation Unit
    - New, better models
    - Random Brilliance

## Included Models (From Image-Super-Resolution)

The following are the models sourced from Image-Super-Resolution

- Super Resolution CNN (SR)
- Expanded Super Resolution CNN (ESR)
- Deep Denoising Super Resolution (DDSR)

There is also an incomplete implementation of *<a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a>*.

- Very Deep Super Resolution (VDSR)


## Credits

*<a href="https://github.com/titu1994/Image-Super-Resolution">Image-Super-Resolution</a>*
for base code.

*https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc* for DPX file io.

**PU: More credits here**
