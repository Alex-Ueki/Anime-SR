# Anime Image Super Resolution using in Keras 2+ (TensorFlow Backend)

Credit to
<i><a href="https://github.com/titu1994/Image-Super-Resolution">Image-Super-Resolution</a></i>
for base code (especially img_utils.py and models.py)

## Usage

<br><b>[1]</b> Open <b>tests.py</b> and un-comment the lines at each section to use a model for training, evaluation, or prediction.
<br><b>[2]</b> Execute tests.py to begin training. GPU is recommended, although if small number of images are provided then GPU may not be required.
<br><b>[3]</b> Training images should be located in train_images. The original SD should be placed in "Alpha", and the HD in "Beta"


## TODOS

<br><b>[1]</b> Have paths be stored in a class object that allows settable values (for border and path directories)
<br><b>[2]</b> Use os.path to have path values that work for both MAC (/) and Windows (\)
<br><b>[3]</b> Implement a function in setup.py that tiles an HD image with a given border size
<br><b>[4]</b> Implement a setup.py main() function that initializes all the directories
<br><b>[5]</b> Implement a function that takes images from input_images and divides them into train_images (for training and validation) and eval_images (for evaluation)

## Model Architecture (From Image-Super-Resolution)

The following are the models sourced from Image-Super-Resolution

<br><b>[1]</b> Super Resolution CNN (SR)
<br><b>[2]</b> Expanded Super Resolution CNN (ESR)
<br><b>[3]</b> Deep Denoiseing Super Resolution (DDSR)

There is also an incomplete implementation of <i><a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a></i>.

<br><b>[1]</b> Very Deep Super Resolution (VDSR)

Note that all models are currently designed to work with 64x64 images.
