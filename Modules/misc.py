"""
Miscellaneous utility functions for Anime-SR
"""

import platform
import os

def clear_screen():
    # Clear the screen.
    os.system( "cls" if platform.system().lower()=="windows" else "clear")
