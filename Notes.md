We do a lot of our video work on Macs so much of this is Mac-centric.

# Thrashing the GPU

If your training or validation set is too large, you can start thrashing the GPU. Things slow down because
stuff has to be constantly moved in and out of GPU memory. One symptom of this is that the rest of your
system becomes very sluggish (Atom, for example, becomes unusable for editing). The solution is to reduce
the size of your training or validation set.

# Theano

This project has only been tested with Keras/Tensorflow. It may work with Theano (there is some code to
transpose images into Theano ordering) but there may be cases that haven't been handled. If you try and
use Theano and things don't work, search in frameops.py and modelio.py for 'theano'.

# The quality setting

The quality option of train.py uses a simple heuristic to rank the tiles in an image by how much detail
they have in them. This helps models train faster because you're not giving them tiles that don't have
much feature information (especially useful with Anime since there are large areas of single-color paint).

Note however that the quality setting (and jitter as well) are not applied to validation images, since
you most definitely do want to know how the model performs on "uninteresting" tiles. Also, you need to
keep this in mind when choosing the number of images in your training and validation sets; if you have
4 times the number of images in your training set but are using quality=0.5, then your real ratio is
2:1 (not counting the extra tiles you may be creating through jittering)

# Jittering

In addition to simply tiling the image, train.py also generates intermediate tiles that are offset
1/2 tile horizontally and vertically, in the hopes of creating additional tiles with interesting
detail. So an NxM tile image, when jittered, will create an additional (N-1)x(M-1) tiles. Jittering
is the default.

# Converting video from interlaced to progressive

Apple Compressor (both old and current versions) have significant issues with reverse telecine on older anime,
because the actual anime is 8 or 12 frames per second, so if there isn't gross cell or background movement,
visible interlace wont happen 2 frames out of 5, but 2 frames out of 10.

JES Deinterlacer seems to work around this issue: https://jeschot.home.xs4all.nl/home.html Also, it seems to generate
output files that are 720x480 or 486 with square pixels, which is what we want to avoid scaling to 640x480 in
Final Cut and Compressor.

- On Input tab, Choose... file to deinterlace, check Top Field first
- On Project tab, select Inverse telecine, check Detect cadence breaks, Output frame rate 23.976
- On Output tab, use Export -> Quicktime Movie and select encoding. Set Size to the specific
size you want, which should match the import size.
- You can also output DPX. It has the same problem as Compressor 4 (see below)

# Creating DPX image files

DPX is a better format than PNG for image files, because the current scipy imread/save routines cannot handle 48 bits/pixel images. Since the standard for uncompressed video is 3x10 = 30 bits/pixel, exporting video as PNG images loses some detail information. DPX can be 10 bits/pixel, and it's a relatively simple format so decoding/encoding it is easy for lazy people like us.

Older version of Compressor (3.5.3) seems to do a better job of Quicktime to DPX conversion than the current Compressor 4.
However, unless you set the output size to match the original input size and pixel aspect ratio (say, 720x486, DV),
it will scale the output DPX images to square pixels (640x486), which is obviously not what you want.

Compressor 4 generates DPX that look weird when viewed in GraphicConverter (as does JES). They have blue bar in the center of the screen and the colors are distorted and clipped. This may be a GraphicConverter issue, but another utility (DJV) also has the same issue.

# DPX to PNG conversion

DPX are (usually) logarithmically-encoded images, as opposed to linearly-encoded bitmap images typically used on computers (ie: PNG). Because of this, when you look at them in many file viewers, they don't look right.

You can use ImageMagick to convert DPX to PNG like this:

```convert {source dpx file} -set colorspace sRGB {destination png file}```

# Miscellaneous ImageMagick

See also some of the shell scripts in Tools for some ImageMagick invocations.

Convert all the 1920x1080 dpx images in a folder into 720x480 dpx images by slicing off the black borders on the left
and right, and then squeezing down from 1440x1080 to 720x480 (without preserving aspect ratio).
Useful for generating training and validation Alpha images when you have HD beta images.

```mogrify -shave 240x0 -resize 720x480\! *.dpx```

# Todos

Old Analog transfers are often going to have edge sharpening artifacts. Need to investigate how to remove these before
upconversion.

What are the best settings for recurrent networks? I am thinking no jitter, shuffle or skip? Should these be the default
for all runs? Or perhaps some way for the model to over-ride?
