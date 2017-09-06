#!/bin/bash
# cd to directory with 60x60 tiles
# shave.sh {first frame number} {last frame number} {frame skip} {file-prefix}
for ((frame=$1; frame<=$2; frame=frame+$3))
do
	fnum=$(printf "%06d" $frame)
	echo $fnum ...
	montage  $4$fnum*.png -depth 16 -tile 24x18 -geometry 60x60+0+0 super-frame-$fnum.png
done
