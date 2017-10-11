import os
import json
import numpy as np

from Modules.misc import oops, terminate, set_docstring, parse_options
from Modules.modelio import ModelIO
import Modules.genomics as genomics
import Modules.models as models
import Modules.frameops as frameops
from keras.utils import plot_model


set_docstring(__doc__)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

print('PUPSR build')
config = ModelIO({'model_type': 'PUPSR'})
model = models.PUPSR(config)

with open(os.path.join('Temp', 'PUPSR.json'), "w") as text_file:
    text_file.write(model.model.to_json())

plot_model(model.model, show_shapes=True, to_file=os.path.join('Temp', 'PUPSR.png'))

print('PUPSR2 build')
config = ModelIO({'model_type': 'PUPSR2'})
model = models.PUPSR2(config)

with open(os.path.join('Temp', 'PUPSR2.json'), "w") as text_file:
    text_file.write(model.model.to_json())

plot_model(model.model, show_shapes=True, to_file=os.path.join('Temp', 'PUPSR2.png'))

print('Genomics build')

model, _ = genomics.build_model('conv_f64_k9_elu-conv_f32_k1_elu-out_k5')

with open(os.path.join('Temp', 'build_model.json'), "w") as text_file:
    text_file.write(model.to_json())

plot_model(model, show_shapes=True, to_file=os.path.join('Temp', 'build_model.png'))

"""
model, _ = genomics.build_model('mod_r3-conv_f64_k9_elu-mod_r2-max_f64_k15_d2_elu-out_k5')

plot_model(model, show_shapes=True, to_file=os.path.join('Temp', 'build_model2.png'))
"""
