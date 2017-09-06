#!/bin/bash
# cd to directory with 64x64 tiles
# shave.sh {first frame number} {last frame number} {frame skip} {path-prefix}
for ((frame=$1; frame<=$2; frame=frame+$3))
do
	fnum=$(printf "%06d" $frame)
	echo $fnum ...
	mogrify -shave 2x2 $4$fnum*.png
done
