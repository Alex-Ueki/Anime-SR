#!/bin/bash
# sdcopy.sh <hd folder> <sd folder> <destination folder>
# for each file in <hd folder>, copies the file of the same name
# in <sd folder> into <destination folder>
#
# Useful for creating training /validation sets with matched files
for image_file in $1/*.png
do
	cp $2/$(basename $image_file) $3/$(basename $image_file)
done
