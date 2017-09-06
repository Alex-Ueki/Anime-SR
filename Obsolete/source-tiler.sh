#!/bin/bash
# cd to directory with 1440x1080 source files
# Must be Tiles directory at same level as "Source" directory
# Generates 64x64 target tiles with 2 pixel overlap on all edges
# (so the unique area is 60x60, matching the target)
# remember to find and remove tiles that are not 64x64
echo "Cleanup..."
cd ../Tiles
for image_file in *.png
do
rm $image_file
done
cd ../Source
echo "Tiling..."
for image_file in *.png
do
convert $image_file -bordercolor black -border 2x2 -crop 24x18+4+4@ +repage +adjoin -depth 16 ../Tiles/$(basename $image_file .png)-%03d.png
done
