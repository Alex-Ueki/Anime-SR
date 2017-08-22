#!/bin/bash
# sdcopy.sh <hd tiles folder> <sd tiles folder> <destination folder>
# for each file in <hd tiles folder>, copies the file of the same name
# in <sd tiles folder> into <destination folder>
for image_file in $1/*.png
do
	cp $2/$(basename $image_file) $3/$(basename $image_file)
done
