# Miscellaneous Notes & Todos

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
- You can also output DPX. I haven't tested it.

Old Analog transfers are often going to have edge sharpening artifacts. Need to investigate how to remove these before
upconversion.

Output of Predict when using 720x486 input is pretty shitty. Need to test and get the scaling working flawlessly.
Also, probably some extra values in DPX header that need to be tweaked to make the output files 100% valid (row width?).
