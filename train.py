"""

Usage: train.py [model] [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    w=nnn      tile width, default=60
    h=nnn      tile height, default=60
    b=nnn      border size, default=2
    e=nnn      epoch size, default=255
    t=path     path to training folder, default = Data/train_images/training
    v=path     path to validation folder, default = Data/train_images/validation
    m=path     path to model file, default = Weights/{model}-{width}-{height}-{border}.h5

"""

import Modules.srcnn as srcnn
import Modules.setup as setup
import Modules.frameops as frameops

import sys
import os

# if is_error is true, display message and optionally end the run.
# return updated error_state

def oops(error_state,is_error,msg,value=0,end_run=False):

    if is_error:
        print('Error: ' + msg.format(value))
        if end_run:
            terminate(True)

    return error_state or is_error

# terminate run if oops errors have been encountered

def terminate(sarah_connor):
    if sarah_connor:
        print("""
Usage: train.py [model] [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    w=nnn      tile width, default=60
    h=nnn      tile height, default=60
    b=nnn      border size, default=2
    e=nnn      epoch size, default=255
    t=path     path to training folder, default = Data/train_images/training
    v=path     path to validation folder, default = Data/train_images/validation
    m=path     path to model file, default = Weights/{model}-{width}-{height}-{border}.h5
""")
        # print("Press ENTER to exit...\n")
        # input()
        sys.exit(1)

if __name__ == "__main__":

    errors = oops(False,len(sys.argv) == 1,"Model type not specified",len(sys.argv)-1,True)

    model_type = sys.argv[1]

    # initialize defaults

    (tile_width, tile_height, tile_border, epochs) = (60, 60, 2, 255)
    paths = { 'training': os.path.abspath(os.path.join('Data','train_images','training')),
              'validation': os.path.abspath(os.path.join('Data','train_images','validation')),
              'weights': os.path.abspath(os.path.join('Data','weights',"%s-%d-%d-%d.h5" % (model_type,tile_width,tile_height,tile_border)))
            }

    for option in sys.argv[2:]:
        opvalue = option.split("=",maxsplit=1)
        if len(opvalue) == 1:
            errors = oops(errors,True,"Unknown option ({})",option)
        else:
            op,value = opvalue
            vnum = int(value) if value.isdigit() else -1

            if op == 'w':
                tile_width = vnum
                errors = oops(errors,vnum <= 0,"Tile width invalid ({})",option)
            elif op == 'h':
                tile_height = vnum
                errors = oops(errors,vnum <= 0,"Tile height invalid ({})",option)
            elif op == 'b':
                tile_border = vnum
                errors = oops(errors,vnum <= 0,"Tile border invalid ({})",option)
            elif op == 'e':
                epochs = vnum
                errors = oops(errors,vnum <= 0,"Epochs invalid ({})",option)
            elif op == "t":
                paths['training'] = os.path.abspath(value)
            elif op == "v":
                paths['validation'] = os.path.abspath(value)
            elif op == "m":
                paths['weights'] = os.path.abspath(value)

    terminate(errors)

    print("        Model : " + model_type)
    print("Source Images : " + paths['training'])
    print("Target Images : " + paths['validation'])
    print("   Model File : " + paths['weights'])
    print("")

    sr = srcnn.BasicSR(base_tile_width=tile_width, base_tile_height=tile_height, border=tile_border, paths=paths)
    sr.create_model()
    sr.fit(nb_epochs=epochs)
    sr.save()

    print("Training completed...")
