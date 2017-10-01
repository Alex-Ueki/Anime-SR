# pylint: disable=C0301
# (line_too_long disabled)
"""
Usage: monitor.py

    Monitors training progress for both evolve.py and train.py by reading and parsing
    .json files.
"""

import re
import os
import json
import datetime

from Modules.misc import set_docstring, clear_screen, BCOLORS

set_docstring(__doc__)

# Maximum display width (in characters)

SWIDTH = 109

# Hilight dashes in genomes If things look garbled, disable. Tested only
# on Windows 10!

DASHIT = True

# Default paths

GENEPOOL = os.path.join('Data', 'genepool.json')
MODELS = os.path.join('Data', 'models')


def fitstr(wide_str, other=0, max_width=SWIDTH):
    """ Truncate or cut the middle out of a string to conform it to
        max_width - other.
    """

    width = max_width - other

    if width < 5:
        return wide_str[:width]

    if len(wide_str) > width:
        width = width - 5
        lpart = width // 2
        rpart = width - lpart
        return wide_str[:lpart] + ' ... ' + wide_str[-rpart:]

    return wide_str


def endash(genome, max_width=0, dashit=DASHIT):
    """ Hilight dash marks and/or right-pad string (which will be a genome). """

    padding = ' ' * max(0, max_width - len(genome))

    if dashit:
        genome = genome.replace('-', BCOLORS.HEADER + '-' + BCOLORS.ENDC)

    return genome + padding


def mod_time(path, suffix='.json'):
    """ Get the last modified time of a path (which may be a folder or file).
        Filter files by suffix.
    """

    if os.path.isfile(path) and path.endswith(suffix):
        return os.path.getmtime(path)

    times = [os.path.getmtime(os.path.join(root, f))
             for (root, _, files) in os.walk(path)
             for f in files if f.endswith(suffix)]

    return max(times) if times else 0.0


def state_files(path, suffix='_state.json'):
    """ Extract the json in all files in the path and return them in a list.
        Filter files by suffix.
    """

    if os.path.isfile(path) and path.endswith(suffix):
        return json.load(open(path, 'r'))

    return [json.load(open(os.path.join(root, f), 'r'))
            for (root, _, files) in os.walk(path)
            for f in files if f.endswith(suffix)]


def genepool_display(data, codon_filter=None, last_mod=None):
    """ Display current status overviews of a genepool.json file """

    def population_display(info, timestamp):
        """ Display the current evolver population """

        info = [p if isinstance(p, list) else [p, 0.0, 0] for p in info]  # legacy format conversion
        info = [[fitstr(p[0], 21), p[1], p[2]] for p in info]

        print('Current Genepool{}\n'.format(timestamp))
        info.sort(key=lambda x: x[1])
        max_width = max([12] + [len(p[0]) for p in info])
        print('{} {:>6s} {:>9s}'.format('Genomes'.ljust(max_width + 4), 'Epochs', 'PSNR'))
        print('{} {:>6s} {:>9s}'.format('-' * (max_width + 4), '-' * 6, '-' * 9))
        for model in info:
            if model[2] > 0:
                print('    {} {:>6d} {:9.4f}'.format(endash(model[0], max_width), model[2], model[1]))
            else:
                print('    {}'.format(endash(model[0], max_width)))

    def graveyard_display(info, timestamp):
        """ Display contents of the graveyard """

        info = [fitstr(p, 4) for p in info]
        max_width = max([15] + [len(p) for p in info])
        print('Genepool graveyard{}\n{}'.format(timestamp, '-' * (max_width + 4)))
        for model in info:
            print('    {}'.format(endash(model)))

    def statistics_display(info, timestamp):
        """" Display evolutionary statistics (with possible filtering) """

        print('Genepool statistics{}\n'.format(timestamp))

        # Gene construction info; cribbed from genomics.py

        acts = ['_linear', '_elu', '_tanh', '_softsign']
        layers = ['conv_', 'add_', 'avg_', 'mult_', 'max_']

        # convert the statistics dictionary to a list of lists

        info = [[fitstr(key, 41)] + info[key] for key in info.keys()]

        # sort by max fitness:mean fitness:worst fitness

        info.sort(key=lambda n: n[1:])

        if codon_filter is None:
            title = 'Complete codons (Count >= 5):'
            codons = [c for c in info if c[4] >= 5 and
                      any(c[0].startswith(x) for x in layers) and
                      any(c[0].endswith(y) for y in acts) and
                      c[0].count('_') > 1]
        else:
            title = 'Filtered codons ({})'.format(codon_filter.pattern)
            codons = [c for c in info if codon_filter.search(c[0])]

        max_width = max([len(title) - 4] + [len(p[0]) for p in codons])
        print('{} {:>9s} {:>9s} {:>9s} {:>6s}'.format(
            title.ljust(max_width + 4), 'Best', 'Mean', 'Worst', 'Count'))
        print('{} {:>9s} {:>9s} {:>9s} {:>6s}'.format(
            '-' * (max_width + 4), '-' * 9, '-' * 9, '-' * 9, '-' * 6))
        for codon in codons:
            print('    {} {:9.4f} {:9.4f} {:9.4f} {:6d}'.format(
                endash(codon[0], max_width), *codon[1:]))

    def io_display(info, timestamp):
        """ Display IO values for evolver """

        for k in info['paths'].keys():
            info['paths[\'{}\']'.format(k)] = info['paths'][k]
        del info['paths']

        print('Genepool IO settings{}\n'.format(timestamp))
        keys = sorted(info.keys())
        max_width = max([len(k) for k in keys])
        for key in sorted(info.keys()):
            print('    {} : {}'.format(key.ljust(max_width), info[key]))

    #---------------------------------
    # genepool_display() function body
    #---------------------------------

    timestamp = '' if last_mod is None else ' (last change {:%Y-%m-%d %I:%M:%S %p})'.format(
        datetime.datetime.fromtimestamp(last_mod))

    with open(GENEPOOL, 'r') as jsonfile:
        state = json.load(jsonfile)

    info = state[data]

    clear_screen()

    select = {'population': population_display,
              'graveyard': graveyard_display,
              'statistics': statistics_display,
              'io': io_display}

    if data in select:
        select[data](info, timestamp)
    else:
        print('Unknown genepool selector [{}]'.format(data))


def models_display(states, last_mod=None):
    """ Display the current models """

    timestamp = '' if last_mod is None else \
                ' (last change {:%Y-%m-%d %I:%M:%S %p})'.format(
                    datetime.datetime.fromtimestamp(last_mod))

    # Gather the info we need from the state files

    info = []
    for state in states:
        bestv = state['best_values']['val_PeakSignaltoNoiseRatio']
        beste = state['best_epoch']['val_PeakSignaltoNoiseRatio']
        ecount = state['epoch_count']
        mname = fitstr(os.path.basename(state['io']['model_path']).split('.h5')[0], 30)
        info.append((bestv, beste, ecount, mname))

    info.sort()

    clear_screen()

    title = 'Trained Models{}'.format(timestamp)

    max_width = max([len(title) - 4] + [len(p[3]) for p in info])
    print('{} {:>9s} {:>7s} {:>7s}'.format(
        title.ljust(max_width + 4), 'Best', '@Epoch', '#Epochs'))
    print('{} {:>9s} {:>7s} {:>7s}'.format(
        '-' * (max_width + 4), '-' * 9, '-' * 7, '-' * 7))
    for item in info:
        print('    {} {:9.4f} {:7d} {:7d}'.format(
            endash(item[3], max_width), item[0], item[1], item[2]))


def monitor():
    """ Monitor the status of training """

    # Config info for the commands. Tuples contain the path to the data the
    # command will display, and the sub-element of the data that will be used

    cmd_info = {'P': (GENEPOOL, 'population'),
                'G': (GENEPOOL, 'graveyard'),
                'S': (GENEPOOL, 'statistics'),
                'F': (GENEPOOL, 'statistics'),
                'I': (GENEPOOL, 'io'),
                'M': (MODELS, None),
                'Q': (GENEPOOL, None)}

    # Default command

    cmd = 'P'
    codon_filter = None

    while True:

        if cmd in cmd_info:

            # Get the last time data we will look at was touched.

            last_mod = mod_time(cmd_info[cmd][0])

            if cmd == 'F':
                print('\nEnter Stats Filter regular expression, then press ENTER.' +
                      ' Leave blank to display best genomes.\n')
                try:
                    codon_filter = input('Filter >')
                    if codon_filter == '':
                        codon_filter = None
                    else:
                        codon_filter = re.compile(codon_filter)
                except EOFError:
                    codon_filter = None
                genepool_display(cmd_info[cmd][1], codon_filter, last_mod)
            elif cmd in ['Q']:
                print('')
                exit(0)
            elif cmd == 'M':
                models_display(state_files(MODELS), last_mod)
            else:
                genepool_display(cmd_info[cmd][1], codon_filter, last_mod)

        # Get user command, then loop back and process it.

        key = input('\n{:%I:%M:%S %p}'.format(datetime.datetime.now()) +
                    ' : M)odels, Genepool P)opulation, G)raveyard,' +
                    ' S)tats, Set F)ilter, I)O, Q)uit [ENTER to refresh] >')

        if key != '':
            cmd = key[0].upper()


if __name__ == '__main__':
    monitor()
