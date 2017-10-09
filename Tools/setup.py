# pylint: disable=C0301
# (line_too_long disabled)
"""
Set up Anime-SR project with default datafile structure
"""

import os


def create_folder(root, folders, depth=0):
    """ Recursively create folder structure from nested list """

    exists = os.path.exists(root)
    if not exists:
        os.makedirs(root)

    print('{}{} {}'.format(' ' * (depth * 4), os.path.abspath(root), '' if exists else ' ** CREATED **'))

    for folder in folders:
        if isinstance(folder, (list, tuple)):
            subroot = os.path.join(root, folder[0])
            create_folder(subroot, folder[1], depth + 1)
        else:
            subroot = os.path.join(root, folder)
            create_folder(subroot, [], depth + 1)

def setup():
    """ Setup folder structure """

    alphabeta = ['Alpha', 'Beta']

    sub_folders = [[s, alphabeta]
                   for s in ['eval_images', 'predict_images']]

    train_images = [[s, alphabeta] for s in ['training', 'validation']]
    sub_folders.append(['train_images', train_images])

    sub_folders.append(['models', ['genes', 'submodels', 'graphs']])

    create_folder('Data', sub_folders, depth=0)

if __name__ == '__main__':
    setup()
