"""
Miscellaneous utility functions for Anime-SR
"""

import platform
import os
import sys
import datetime

# Will be set by module that imports us - used by terminate() to display
# appropriate help.

_DOCSTRING = ''


def set_docstring(docstring):
    """ Set the stored docstring; called by outermost scope """

    global _DOCSTRING
    _DOCSTRING = docstring


def clear_screen():
    """ Clear the screen """

    os.system('cls' if platform.system().lower() == 'windows' else 'clear')


def printlog(*s):
    """ Print with timestamp """

    print('{:%Y-%m-%d %H:%M:%S.%f}:'.format(datetime.datetime.now()), *s)


""" ANSI terminal color escapes
    https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
"""


class BCOLORS():
    """ List of ANSI terminal color escapes """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup_default(paths):
    """ Create default directories -- PU not needed? """

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def oops(error_state, is_error, msg, error_value=None, end_run=False):
    """ If is_error is true, display message and optionally end the run.
        return updated error_state. error_value may be a tuple of
        arguments for format().
    """

    if is_error:

        if error_value is not None:
            if isinstance(error_value, (list, tuple)):
                msg = msg.format(*error_value)
            else:
                msg = msg.format(error_value)

        print('Error: ' + msg)

        if end_run:
            terminate(True)

    return error_state or is_error


def validate(error_state, new_value, is_error, msg, error_value=None, end_run=False):
    """ Validate input, return a new_value, error tuple. Since we will never use the
        new_value if error_state ever becomes True, it's ok to blindly return it.

        error_state   Current validation error state
        new_value     The new value of the current option
        is_error      Is the new_value of the current option bad?
        msg           Error message format() string
        error_value   Value to display in error message; may be tuple.
                      (if None, use the new_value)
        end_run       Immediately terminate if we get an error?
    """

    if is_error:
        if error_value is None:
            error_value = new_value
        error_state = oops(error_state, is_error, msg, error_value, end_run)

    return (new_value, error_state)


def terminate(sarah_connor, verbose=True):
    """ Terminate run if errors have been encountered.
        Parental Unit has already done penance for this joke.
    """

    if sarah_connor:
        if verbose:
            print(_DOCSTRING)
        print('')
        sys.exit(1)


def opcheck(option, oldvalue, newvalue, errors):
    """ Check option for validity, return new value if OK. Option is a tuple of
        (config key, type, validation function (if true, error!), error string)
    """

    # Convert type of newvalue.

    original_value = newvalue

    if option[1] == str:
        newvalue = str(newvalue)
    elif option[1] == bool:
        newvalue = True if 'true'.startswith(newvalue.lower()) else newvalue
        newvalue = False if 'false'.startswith(newvalue.lower()) else newvalue
        try:
            newvalue = bool(newvalue)
        except ValueError:
            newvalue = -1
    elif option[1] == int:
        try:
            newvalue = int(newvalue)
        except ValueError:
            newvalue = -1
    elif option[1] == float:
        try:
            newvalue = float(newvalue)
        except ValueError:
            newvalue = -1.0

    if option[2](newvalue):
        errors = True
        print(option[3].format(original_value))
        newvalue = oldvalue

    return (errors, newvalue)


def parse_options(opcodes):
    """ Parse options. Takes a dictionary of options, each element is a tuple
        containing 4 elements:

        config_name     the ModelIO element name
        type            type of the option
        invalid_func    func that returns True if option is out of bounds
        format_string   message to display if there is a problem with {}
    """

    options = {}

    # Option names must be sorted because some options can be substrings of others

    option_names = sorted(list(opcodes.keys()))

    errors = False

    for param in sys.argv[1:]:

        opvalue = param.split('=', maxsplit=1)

        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', param)
            continue

        option, value = opvalue[0].lower(), opvalue[1]

        # Match option, make sure it isn't ambiguous.

        opmatch = [s for s in option_names if s.startswith(option)]

        if not opmatch or len(opmatch) > 1 and opmatch[0] != option:
            errors = oops(errors, True, '{} option ({})',
                          ('Ambiguous' if opmatch else 'Unknown', option))
            continue

        opcode = opcodes[opmatch[0]]
        opname = opcode[0]

        if opname not in options:
            options[opname] = None

        errors, options[opname] = opcheck(opcode, options[opname], value, errors)

    terminate(errors)

    # Move _paths temp configs into options['paths']

    temp_paths = [path for path in options if path.endswith('_path')]
    options['paths'] = {}
    for path in temp_paths:
        options['paths'][path.split('_path')[0]] = options[path]
        del options[path]

    return options
