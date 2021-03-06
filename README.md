# Anime Image Super Resolution using Keras (TensorFlow Backend)

Toolkit for investigating techniques for improved upconversion of SD video to HD. Can potentially be adapted for
use on other image conversion tasks.

## Features

- Tiling of images with overlapping borders.
- Quality-culling of tiles based on approximate image detail.
- Tools for data preparation, training, evaluation, prediction, genetic evolution of models.
- Training can be interrupted and restarted freely.

## Performance

Performance Metric for this framework is [Peak Signal-to-Noise ratio](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)      
Images drawn from the Riding Bean anime, where each training pair is a naively upscaled SD image with a true HD image.
Models were trained to predict either full images or residual images (the difference between the SD and HD)

Training Set: 66 pairs of 1920x1080 .dpx Frames, tiled into 60x60 segments with 2-pixel borders     
Validation Set: 18 pairs of 1920x1080 .dpx Frames, tiled into 60x60 segments with 2-pixel borders       
Evaluation Set: 18 pairs of 1920x1080 .dpx Frames, tiled into 60x60 segments with 2-pixel borders   

All models were trained on 1612 batches of size 16 per epoch

Results: After 100 epochs
- BasicSR: PeekSignalToNoiseRatio = 45.55115
- BasicSR w/ Residual: PeekSignalToNoiseRatio = 50.00968
- DeepDenoiseSR: PeekSignalToNoiseRatio = 48.34253
- DeepDenoiseSR w/ Residual: PeekSignalToNoiseRatio = 50.83483
- VDSR: PeekSignalToNoiseRatio = 20.63989 (Probably an issue with current implementation)
- VDSR w/ Residual: PeekSignalToNoiseRatio = 50.35691
- VDSR w/ ELU: PeekSignalToNoiseRatio = 40.31843
- VDSR w/ ELU & Residual: PeekSignalToNoiseRatio = 50.78876

## Setup

Requires python 3.5+, Keras, assorted standard packages (numpy, scipy, etc.)

Data directory should be as follows (Tools/setup.py will do this for you)

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
    models/                 .h5 and .json files for trained models.
```

See *Notes.md* for useful tips / tricks / comments

## Tool Usage


#### train.py [option(s)] ...
```
Trains a model. See Modules/models.py for sample model types.

Options are:

    type=model          model type, default is BasicSR. Model type can also be a genome (see below)
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          max epoch count, default=100. See below for more details
    epochs+=nnn         run epoch count, default=None. Overrides epochs=nnn
    lr=.nnn             set initial learning rate, default = use model's current learning rate. Should be 0.001 or less
    quality=.nnn        fraction of the "best" tiles used in training. Default is 1.0 (use all tiles)
    residual=1|0|T|F    have the model train using residual images. Default is false.
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240
    trimright=nnn       pixels to trim on image right edge, default = 240
    trimtop=nnn         pixels to trim on image top edge, default = 0
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0
    jitter=1|0|T|F      include jittered tiles (offset by half a tile across&down) when training; default=False
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=False
    shuffle=1|0|T|F     shuffle tiles into random order when training; default=True
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to trained model file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}.h5
    state=path          path to state file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}.json
    verbose=1|0|T|F     display verbose training output, default = True
    bargraph=1|0|T|F    display bargraph of training progress, default = True

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK).

    You can terminate a training session with ^C, and then resume training by reissuing the same command. The model's state
    is completely stored in the .h5 file, and the training state is in the .json file.

    The epochs value is the maximum number of epochs that will be trained **over multiple sessions**. So if you have
    previously trained a model for 50 epochs, epochs=75 would mean the model trains for 25 additional epochs. Alternately,
    you could specify epochs+=25 to limit the current training run to 25 epochs.

    If the type is not one of the standard defined models, then it is checked to see if it is a valid genome.
    If so, the model and state files (if not set) are searched for in {Data}/models and {Data}/models/genes
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

    Expects that there will be a matching .json file for the model
    (ie: BasicSR-60-60-2-dpx.json) that contains all the tiling/trimming information
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
    test=1|0|T|F        If true, only predict first, last and middle image. Default=False
    png=1|0|T|F         If true, force output images to be png. Default=False
    diff=1|0|T|F        If true, generate input/output difference images. Default=False
                        Generates regular (-diff) and normalized (-ndiff) images.

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching .json file for the model
    (ie: BasicSR-60-60-2-dpx.json) that contains all the tiling/trimming information
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
    jitter=1|0|T|F      include jittered tiles (offset by half a tile across&down) when training; default=False
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=False
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
#### Usage: graph.py [option(s)] ...
````
    Generates nice graph of a model.

Options are:

    data=path           path to the main data folder, default = Data
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5
    graph=path          output file path. Default = {Data}/models/graphs/{model}.png

    Expects that there will be a matching .json file for the model
    (ie: BasicSR-60-60-2-dpx.json) that contains all the tiling/trimming information
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

- Leverage the video aspect of the images to improve results (recurrent models)
- Add in the latest fads and models, e.g Capsule Networks
- Look into gaussian output layers for residual models to boost convergence speeds (assuming residuals are near zero)
- Determine the optimal training time for convergence (should be 100+ epochs).

## Included Models (From Image-Super-Resolution)

The following are the models sourced from Image-Super-Resolution

- Super Resolution CNN (SR)
- Expanded Super Resolution CNN (ESR)
- Deep Denoising Super Resolution (DDSR)

From *<a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a>*.
- Very Deep Super Resolution (VDSR)

Additionally, as a modification
- Exponential Linear Unit Variations on the above models

Finally, there are models that can be generated using evolve.py

## Credits

*<a href="https://github.com/titu1994/Image-Super-Resolution">Image-Super-Resolution</a>*
for base code and implementations of     
Super Resolution CNN, Expanded Super Resolution, Deep Denoising Super Resolution.

*<a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a>*.
for the model structure of VDSR.

*https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc* for DPX file io.
