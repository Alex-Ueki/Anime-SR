#!/bin/bash
# cd to directory with 1440x1080 source files
# Must be Tiles directory at same level as source directory
# Execute script to create tiles for all images
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
convert $image_file -crop 64x64 +repage +adjoin ../Tiles/$image_file-%03d.png
done
# at this point we have created regular, FLOP- and FLIP- versions of the tiles.
# now we flip and flop the tiles so they are in correct orientation.
echo "Flip/Flop..."
cd ../Tiles
for image_file in FLOP-*.png
do
mogrify -flop $image_file
done
for image_file in FLIP-*.png
do
mogrify -flip $image_file
done
cd ../Source
