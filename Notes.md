Miscellaneous notes

Apple Compressor (both old and current versions) have significant issues with reverse telecine on older anime,
because the actual anime is 8 or 12 frames per second, so if there isn't gross cell or background movement,
visible interlace wont happen 2 frames out of 5, but 2 frames out of 10.

JES Deinterlacer seems to work around this issue: https://jeschot.home.xs4all.nl/home.html

Old Analog transfers are often going to have edge sharpening artifacts. Need to investigate how to remove these before
upconversion.

Need to document good DPX generation workflow for 720x480; there are issues with Final Cut scaling to 640x480 (because it
wants the pixels to be square)
