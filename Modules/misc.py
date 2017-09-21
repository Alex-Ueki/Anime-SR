"""
Miscellaneous utility functions for Anime-SR
"""

import platform
import os
import sys

# Will be set by module that imports us

owner_docstring = ''

def set_docstring(docstring):

    global owner_docstring
    owner_docstring = docstring

# Clear the screen


def clear_screen():
    # Clear the screen.
    os.system('cls' if platform.system().lower() == 'windows' else 'clear')

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
