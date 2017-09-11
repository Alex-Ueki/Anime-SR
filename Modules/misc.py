"""
Miscellaneous utility functions for Anime-SR
"""

import platform
import os
import frameops
import numpy as np

def clear_screen():
    # Clear the screen.
    os.system( "cls" if platform.system().lower()=="windows" else "clear")

def setup_default(paths):
    for key, value in path.iteritems():
        if not os.path.exists(value):
            os.makedirs(value)
