"""
Miscellaneous utility functions for Anime-SR
"""

import platform
import os
import sys
import datetime
import multiprocessing
import time

# Will be set by module that imports us

owner_docstring = ''


def set_docstring(docstring):

    global owner_docstring
    owner_docstring = docstring

# Clear the screen


def clear_screen():

    os.system('cls' if platform.system().lower() == 'windows' else 'clear')

# Print with timestamp


def printlog(*s):

    print('{:%Y-%m-%d %H:%M:%S.%f}:'.format(datetime.datetime.now()), *s)

# Setup default directories


def setup_default(paths):
    for key, value in path.iteritems():
        if not os.path.exists(value):
            os.makedirs(value)

# If is_error is true, display message and optionally end the run.
# return updated error_state. error_value may be a tuple of
# arguments for format().


def oops(error_state, is_error, msg, error_value=0, end_run=False):

    if is_error:
        # Have to handle the single/multiple argument case.
        # if we pass format a simple string using *value it
        # gets treated as a list of individual characters.

        print('Error: ' + (msg.format(*error_value) if type(error_value) in (list, tuple) else
                           msg.format(error_value)))

        if end_run:
            terminate(True)

    return error_state or is_error

# Validate input, return a new_value, error tuple. Since we will never use the
# new_value if error_state ever becomes True, it's ok to blindly return it.
#
# error_state   Current validation error state
# new_value     The new value of the current option
# is_error      Is the new_value of the current option bad?
# msg           Error message format() string
# error_value   Value to display in error message; may be tuple.
#               (if None, use the new_value)
# end_run       Immediately terminate if we get an error?


def validate(error_state, new_value, is_error, msg, error_value=None, end_run=False):

    if is_error:
        if error_value == None:
            error_value = new_value
        error_state = oops(error_state, is_error, msg, error_value, end_run)

    return (new_value, error_state)


# Terminate run if errors have been encountered.
# Parental Unit has already done penance for this pun.


def terminate(sarah_connor, verbose=True):
    if sarah_connor:
        if verbose:
            print(owner_docstring)
        print('')
        sys.exit(1)

# Single character input routine - gets single character from std input without requiring
# a Newline. See: https://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user


def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch

getch = _find_getch()

# Version of getch with timeout (returns byte \x00 if timeout triggered). See:
# https://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
# https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread-in-python

def _input(q):

    q.put(getch())
    return

def _slp(tm, q):

    time.sleep(tm)
    q.put(b'\x00')
    return

def getch_with_timeout(timeout=30):

    q = multiprocessing.Queue()
    th = multiprocessing.Process(target=_input, args=(q,))
    tt = multiprocessing.Process(target=_slp, args=(timeout, q,))

    th.start()
    tt.start()

    ret = q.get()
    th.terminate()
    tt.terminate()

    return ret
