"""
Usage: monitor.py

    Monitors training progress for both evolve.py and train.py by reading and parsing
    .json files.
"""

from Modules.misc import set_docstring, printlog, clear_screen, bcolors
from functools import reduce

import sys
import re
import os
import json
import time
import datetime
import itertools

set_docstring(__doc__)

# Maximum display width (in characters)

SWIDTH = 109

# Hilight dashes in genomes If things look garbled, disable. Tested only
# on Windows 10!

DASHIT = True

# Default paths

genepool = os.path.join('Data', 'genepool.json')
models = os.path.join('Data', 'models')


def fitstr(s, other):

    w = SWIDTH - other

    if w < 5:
        return s[:w]
    elif len(s) > w:
        w = w - 5
        lc = w // 2
        rc = w - lc
        return s[:lc] + ' ... ' + s[-rc:]
    else:
        return s

# Hilight dash marks and optionally right-pad


def endash(s, max_width=0):

    padding = max(0, max_width - len(s))

    if DASHIT:
        s = s.replace('-', bcolors.HEADER + '-' + bcolors.ENDC)

    return s + ' ' * padding

# Get the last modified time of a path (which may be a folder or None).
# Filter files by suffix.


def mod_time(path, suffix='.json'):

    if os.path.isfile(path) and path.endswith(suffix):
        return os.path.getmtime(path)
    else:
        times = 0.0
        for (root, dirs, files) in os.walk(path):
            times = max([times] + [os.path.getmtime(os.path.join(root, f))
                                   for f in files if f.endswith(suffix)])
        return times

# Extract the contents of all the _state.json files in the path and return
# them in a list.


def state_files(path, suffix='_state.json'):

    if os.path.isfile(path) and path.endswith(suffix):
        return json.load(open(path, 'r'))
    else:
        jsons = []
        for (root, dirs, files) in os.walk(path):
            jsons.extend([json.load(open(os.path.join(root, f), 'r'))
                          for f in files if f.endswith(suffix)])
        return jsons

# Display current status overviews of a genepool.json file


def genepool_display(data, filter=None, last_mod=None):


    timestamp = '' if last_mod == None else ' (last change {:%Y-%m-%d %I:%M:%S %p})'.format(
        datetime.datetime.fromtimestamp(last_mod))

    with open(genepool, 'r') as f:
        state = json.load(f)

    info = state[data]

    clear_screen()

    if data == 'population':

        trained = [[fitstr(p[0], 14), p[1]] for p in info if type(p) is list]
        untrained = [fitstr(p, 4) for p in info if type(p) is not list]

        print('Current Population{}\n'.format(timestamp))
        trained.sort(key=lambda x: x[1])
        max_width = max([12] + [len(p[0]) for p in trained])
        print('{} {:>9s}'.format('Trained Genomes:'.ljust(max_width + 4), 'PSNR'))
        print('{} {:>9s}'.format('-' * (max_width + 4), '-' * 9))
        for model in trained:

            print('    {} {:9.4f}'.format(
                endash(model[0], max_width), model[1]))

        if len(untrained) > 0:
            max_width = max([len(p) for p in untrained])
            print('\nUntrained Genomes:\n{}'.format('-' * (max_width + 4)))
            for model in untrained:
                print('    {}'.format(endash(model)))

    elif data == 'graveyard':

        info = [fitstr(p, 4) for p in info]
        max_width = max([15] + [len(p) for p in info])
        print('Genepool graveyard{}\n{}'.format(
            timestamp, '-' * (max_width + 4)))
        for model in info:
            print('    {}'.format(endash(model)))

    elif data == 'statistics':
        print('Genepool statistics{}\n'.format(timestamp))

        # Gene construction info; cribbed from genomics.py

        acts = ['_linear', '_elu', '_tanh', '_softsign']
        layers = ['conv_', 'add_', 'avg_', 'mult_', 'max_']

        # convert the statistics dictionary to a list of lists

        info = [[fitstr(key, 41)] + info[key] for key in info.keys()]

        # sort by max fitness:mean fitness:worst fitness

        info.sort(key=lambda n: n[1:])

        if codon_filter == None:
            title = 'Complete codons (Count >= 5):'
            codons = [c for c in info if c[4] >= 5 and any(c[0].startswith(
                x) for x in layers) and any(c[0].endswith(y) for y in acts) and c[0].count('_') > 1]
        else:
            title = 'Filtered codons ({})'.format(codon_filter.pattern)
            codons = [c for c in info if codon_filter.search(c[0])]

        max_width = max([len(title) - 4] + [len(p[0]) for p in codons])
        print('{} {:>9s} {:>9s} {:>9s} {:>6s}'.format(
            title.ljust(max_width + 4), 'Best', 'Mean', 'Worst', 'Count'))
        d9 = '-' * 9
        print('{} {:>9s} {:>9s} {:>9s} {:>6s}'.format(
            '-' * (max_width + 4), d9, d9, d9, '-' * 6))
        for codon in codons:
            print('    {} {:9.4f} {:9.4f} {:9.4f} {:6d}'.format(
                endash(codon[0], max_width), codon[1], codon[2], codon[3], codon[4]))

    elif data == 'io':

        for k in info['paths'].keys():
            info['paths[\'{}\']'.format(k)] = info['paths'][k]
        del info['paths']

        print('Genepool IO settings{}\n'.format(timestamp))
        keys = sorted(info.keys())
        max_width = max([len(k) for k in keys])
        for key in sorted(info.keys()):
            print('    {} : {}'.format(key.ljust(max_width), info[key]))

    else:
        print('Unknown genepool selector [{}]'.format(data))


def models_display(states, last_mod=None):

    timestamp = '' if last_mod == None else ' (last change {:%Y-%m-%d %I:%M:%S %p})'.format(
        datetime.datetime.fromtimestamp(last_mod))

    # Gather the info we need from the state files

    info = []
    for s in states:
        bv = s['best_values']['val_PeakSignaltoNoiseRatio']
        be = s['best_epoch']['val_PeakSignaltoNoiseRatio']
        ep = s['epoch_count']
        sp = fitstr(os.path.basename(
            s['io']['model_path']).split('.h5')[0], 30)
        info.append((bv, be, ep, sp))

    info.sort()

    clear_screen()

    title = 'Trained Models{}'.format(timestamp)

    max_width = max([len(title) - 4] + [len(p[3]) for p in info])
    print('{} {:>9s} {:>7s} {:>7s}'.format(
        title.ljust(max_width + 4), 'Best', '@Epoch', '#Epochs'))
    d9 = '-' * 9
    print('{} {:>9s} {:>7s} {:>7s}'.format(
        '-' * (max_width + 4), '-' * 9, '-' * 7, '-' * 7))
    for item in info:
        print('    {} {:9.4f} {:7d} {:7d}'.format(
            endash(item[3], max_width), item[0], item[1], item[2]))


if __name__ == '__main__':

    # Paths to monitor for changes, based on the last command displayed.
    # Currently not used

    monitor_paths = {'P': genepool,
                     'G': genepool,
                     'S': genepool,
                     'F': genepool,
                     'I': genepool,
                     'M': models,
                     }

    # Genepool.json elements to monitor, based on display requested

    genepool_data = {'P': 'population',
                     'G': 'graveyard',
                     'S': 'statistics',
                     'I': 'io',
                     }

    # Default command

    cmd = 'P'
    codon_filter = None

    while True:

        # Execute user command

        last_mod = mod_time(
            monitor_paths[cmd]) if cmd in monitor_paths else 10000000.0

        if cmd in genepool_data:
            genepool_display(genepool_data[cmd], codon_filter, last_mod)
        elif cmd == 'M':
            models_display(state_files(models), last_mod)
        elif cmd == 'F':
            print(
                '\nEnter Stats Filter regular expression, then press ENTER. Leave blank to display best genomes.\n')
            try:
                codon_filter = input('Filter >')
                if codon_filter == '':
                    codon_filter = None
                else:
                    codon_filter = re.compile(codon_filter)
            except EOFError:
                codon_filter = None
            cmd = 'S'
            genepool_display(genepool_data[cmd], codon_filter, last_mod)
        elif cmd in ['Q']:
            print('')
            exit(0)

        # Get user command, then loop back and process it.

        ch = input('\n{:%I:%M:%S %p} : M)odels, Genepool P)opulation, G)raveyard, S)tats, Set F)ilter, I)O, Q)uit [ENTER to refresh] >'.format(
            datetime.datetime.now()))

        if ch != '':
            cmd = ch[0].upper()
