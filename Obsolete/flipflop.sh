#!/bin/bash
# Creates FLOP- and FLIP- variants of source files; deletes old versions
# cd to directory with 1440x1080 source files
rm FLOP-*.png
rm FLIP-*.png
for image_file in *.png
do
convert $image_file -flop FLOP-$image_file
convert $image_file -flip FLIP-$image_file
done
#