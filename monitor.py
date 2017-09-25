"""
Usage: monitor.py

    Monitors training progress for both evolve.py and train.py by reading and parsing
    .json files.
"""

from Modules.misc import oops, validate, terminate, set_docstring, printlog, clear_screen
from functools import reduce

import sys
import os
import json
import time
import datetime
import itertools

set_docstring(__doc__)

# Paths

genepool = os.path.join('Data', 'genepool.json')
models = os.path.join('Data', 'models')

# Gene construction info; cribbed from genomics.py

filters = [32, 64, 128, 256]
kernels = [1, 3, 5, 7, 9]
depths = [2, 3, 4]
acts = ['linear', 'elu', 'tanh', 'softsign']
layers = ['conv', 'add', 'avg', 'mult', 'max']

# Get the last modified time of a path (which may be a folder or None).
# Filter files by suffix.

def mod_time(path, suffix='.json'):

    if os.path.isfile(path) and path.endswith(suffix):
        return os.path.getmtime(path)
    else:
        times = 0.0
        for (root, dirs, files) in os.walk(path):
            print([os.path.getmtime(os.path.join(root, f)) for f in files if f.endswith(suffix)])
            times = max([times] + [os.path.getmtime(os.path.join(root, f)) for f in files if f.endswith(suffix)])
        return times

# Extract the contents of all the _state.json files in the path and return
# them in a list.

def state_files(path, suffix='_state.json'):

    if os.path.isfile(path) and path.endswith(suffix):
        return json.load(open(path, 'r'))
    else:
        jsons = []
        for (root, dirs, files) in os.walk(path):
            print(root, dirs, files)
            jsons.extend([json.load(open(os.path.join(root, f), 'r')) for f in files if f.endswith(suffix)])
        return jsons

# Display current status overviews of a genepool.json file


def genepool_display(data, filter=None, last_mod=None):

    if last_mod == None:
        timestamp = ''
    else:
        timestamp = ' (last change {:%Y-%m-%d %I:%M:%S %p})'.format(datetime.datetime.fromtimestamp(last_mod))

    with open(genepool, 'r') as f:
        state = json.load(f)

    info = state[data]

    clear_screen()

    if data == 'population':

        trained = [p for p in info if type(p) is list]
        untrained = [p for p in info if type(p) is not list]

        print('Current Population{}\n'.format(timestamp))
        trained.sort(key=lambda x: x[1])
        max_width = max([12] + [len(p[0]) for p in trained])
        print('{} {:>9s}'.format('Trained Genomes:'.ljust(max_width + 4), 'PSNR'))
        print('{} {:>9s}'.format('-' * (max_width + 4), '-' * 9))
        for model in trained:
            print('    {} {:9.4f}'.format(
                model[0].ljust(max_width), model[1]))

        if len(untrained) > 0:
            max_width = max([len(p) for p in untrained])
            print('\nUntrained Genomes:\n{}'.format('-' * (max_width + 4)))
            for model in untrained:
                print('    {}'.format(model))

    elif data == 'graveyard':

        max_width = max([15] + [len(p) for p in info])
        print('Genepool graveyard{}\n{}'.format(timestamp, '-' * (max_width + 4)))
        for model in info:
            print('    {}'.format(model))

    elif data == 'statistics':
        # Gene construction info; cribbed from genomics.py

        print('Genepool statistics{}\n'.format(timestamp))

        acts = ['_linear', '_elu', '_tanh', '_softsign']
        layers = ['conv_', 'add_', 'avg_', 'mult_', 'max_']

        # convert the statistics dictionary to a list of lists

        info = [[key] + info[key] for key in info.keys()]

        # sort by max fitness

        info.sort(key=lambda n: n[1])

        if codon_filter == None:
            title = 'Complete codons (Count >= 5):'
            codons = [c for c in info if c[4] >= 5 and any(c[0].startswith(
                x) for x in layers) and any(c[0].endswith(y) for y in acts) and c[0].count('_') > 1]
        else:
            title = 'Filtered codons ({})'.format(','.join(codon_filter))
            codons = [c for c in info if all(s in c[0] for s in codon_filter)]

        max_width = max([len(title)-4] + [len(p[0]) for p in codons])
        print('{} {:>9s} {:>9s} {:>9s} {:>6s}'.format(
            title.ljust(max_width + 4), 'Best', 'Median', 'Worst', 'Count'))
        d9 = '-' * 9
        print('{} {:>9s} {:>9s} {:>9s} {:>6s}'.format('-' * (max_width + 4), d9, d9, d9, '-' * 6))
        for codon in codons:
            print('    {} {:9.4f} {:9.4f} {:9.4f} {:6d}'.format(
                codon[0].ljust(max_width), codon[1], codon[2], codon[3], codon[4]))


    elif data == 'io':

        print('Genepool IO settings{}\n'.format(timestamp))
        keys = sorted(info.keys())
        max_width = max([len(k) for k in keys])
        for key in sorted(info.keys()):
            print('    {} : {}'.format(key.ljust(max_width), info[key]))

    else:
        print('Unknown genepool selector [{}]'.format(data))

def models_display(states, last_mod=None):

    if last_mod == None:
        timestamp = ''
    else:
        timestamp = ' (last change {:%Y-%m-%d %I:%M:%S %p})'.format(datetime.datetime.fromtimestamp(last_mod))

    # Gather the info we need from the state files

    info = []
    for s in states:
        bv = s['best_values']['val_PeakSignaltoNoiseRatio']
        be = s['best_epoch']['val_PeakSignaltoNoiseRatio']
        ep = s['epoch_count']
        sp = os.path.basename(s['io']['model_path']).split('.h5')[0]
        info.append((bv, be, ep, sp))

    info.sort()

    clear_screen()

    title = 'Trained Models{}'.format(timestamp)

    max_width = max([len(title)-4] + [len(p[3]) for p in info])
    print('{} {:>9s} {:>7s} {:>7s}'.format(
        title.ljust(max_width + 4), 'Best', '@Epoch', '#Epochs'))
    d9 = '-' * 9
    print('{} {:>9s} {:>7s} {:>7s}'.format('-' * (max_width + 4), '-' * 9, '-' * 7, '-' * 7))
    for item in info:
        print('    {} {:9.4f} {:7d} {:7d}'.format(
            item[3].ljust(max_width), item[0], item[1], item[2]))

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

        # Display requested information

        last_mod = mod_time(monitor_paths[cmd]) if cmd in monitor_paths else 10000000.0

        if cmd in genepool_data:
            genepool_display(genepool_data[cmd], codon_filter, last_mod)
        elif cmd == 'M':
            models_display(state_files(models), last_mod)
        elif cmd == 'F':
            print('\nEnter Stats Filter (one or more substrings, comma-separated), then press ENTER.')
            print('An empty filter displays the best complete codons.\n')
            try:
                codon_filter = input('Filter >')
            except EOFError:
                codon_filter = ''
            print(codon_filter)
            codon_filter = None if codon_filter == '' else codon_filter.split(',')
            cmd = 'S'
            genepool_display(genepool_data[cmd], codon_filter, last_mod)
        elif cmd in ['Q']:
            print('')
            exit(0)

        # Get user command

        ch = input('\n{:%I:%M:%S %p} : M)odels, Genepool P)opulation, G)raveyard, S)tats, Set F)ilter, I)O, Q)uit [ENTER to refresh] >'.format(
                datetime.datetime.now()))

        if ch != '':
            cmd = ch[0].upper()
