#!/bin/bash
# cd to directory with 1440x1080 target files
# Must be Tiles directory at same level as "Target" directory
# Generates 60x60 target tiles
echo "Cleanup..."
cd ../Tiles
for image_file in *.png
do
rm $image_file
done
cd ../Target
echo "Tiling..."
for image_file in *.png
do
convert $image_file -crop 24x18@ +repage +adjoin ../Tiles/$(basename $image_file .png)-%03d.png
done
