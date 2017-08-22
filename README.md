# Anime Image Super Resolution using in Keras 2+ (TensorFlow Backend)

Credit to
<i><a href="https://github.com/titu1994/Image-Super-Resolution">Image-Super-Resolution</a></i>.
for base code (especially img_utils.py and models.py)

## Usage

<br><b>[1]</b> Open <b>tests.py</b> and un-comment the lines at model.fit(...), where model can be sr, esr or dsr, ddsr.
<br><b>[2]</b> Execute tests.py to begin training. GPU is recommended, although if small number of images are provided then GPU may not be required.
<br><b>[3]</b> Training images should be located in train_images. The original SD should be placed in "Alpha", and the HD in "Beta"

###python test.py


## TODOS

## Model Architecture (From Image-Super-Resolution)

### Super Resolution CNN (SRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/SRCNN.png" height=100% width=25%>

The model above is the simplest model of the ones described in the paper above, consisting of the 9-1-5 model.
Larger architectures can be easily made, but come at the cost of execution time, especially on CPU.

However there are some differences from the original paper:
<br><b>[1]</b> Used the Adam optimizer instead of RMSProp.
<br><b>[2]</b> This model contains some 21,000 parameters, more than the 8,400 of the original paper.

It is to be noted that the original models underperform compared to the results posted in the paper. This may be due to the only 91 images being the training set compared to the entire ILSVR 2013 image set. It still performs well, however images are slightly noisy.

### Expanded Super Resolution CNN (ESRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/ESRCNN.png" height=100% width=75%>

The above is called "Expanded SRCNN", which performs slightly worse than the default SRCNN model on Set5 (PSNR 31.78 dB vs 32.4 dB).

The "Expansion" occurs in the intermediate hidden layer, in which instead of just 1x1 kernels, we also use 3x3 and 5x5 kernels in order to maximize information learned from the layer. The outputs of this layer are then averaged, in order to construct more robust upscaled images.

### Deep Denoiseing Super Resolution (DDSRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/Deep Denoise.png" height=100% width=40%>

The above is the "Deep Denoiseing SRCNN", which is a modified form of the architecture described in the paper <a href="http://arxiv.org/abs/1606.08921">"Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections"</a> applied to image super-resolution. It can perform far better than even the Denoiseing SRCNN, but is currently not working properly.

Similar to the paper <a href="http://arxiv.org/abs/1606.08921">Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections</a>, this can be considered a highly simplified and shallow model compared to the 30 layer architecture used in the above paper.

<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/DDSRCNN%20validation%20plot.png" width=100% height=100%>

### Very Deep Super Resolution (VDSRCNN)

Incomplete implementation of <i><a href="https://arxiv.org/abs/1511.04587">Accurate Image Super-Resolution Using Very Deep Convolutional Networks</a></i>.
