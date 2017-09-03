"""
Set up Anime-SR project with default datafile structure
"""

import os

def create_folder(root, folders):

    if os.path.exists(root):
        print('Exists ', root)
    else:
        print('Creating ', root)
        os.makedirs(root)

    for f in folders:
        if type(f) in (list, tuple):
            subroot = os.path.join(root,f[0])
            if len(f) > 1:
                create_folder(subroot, f[1])
        else:
            subroot = os.path.join(root,f)
            create_folder(subroot,[])


if __name__ == '__main__':

    ab = ['Alpha', 'Beta']

    sub_folders = [ [s, ab] for s in ['eval_images', 'input_images', 'predict_images'] ]

    train_images = [ [s, ab] for s in ['training', 'validation']]
    sub_folders.append(['train_images', train_images])

    sub_folders.append(['weights'])

    print(sub_folders)

    create_folder('Data', sub_folders)
