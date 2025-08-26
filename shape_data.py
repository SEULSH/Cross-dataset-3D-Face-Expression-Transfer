
### Code obtained and modified from https://github.com/anuragranj/coma, Copyright (c) 2018 Anurag Ranjan, Timo Bolkart, Soubhik Sanyal, Michael J. Black and the Max Planck Gesellschaft

import os
import numpy as np
from psbody.mesh import Mesh

# try:
#     import psbody.mesh
#     found = True
# except ImportError:
#     found = False
# if found:
#     from psbody.mesh import Mesh

from trimesh.exchange.export import export_mesh
import trimesh

import time
from tqdm import tqdm


class ShapeData(object):
    def __init__(self, datapath, normalization, template_file, meshpackage):

        self.datapath = datapath

        self.normalization = normalization

        self.meshpackage = meshpackage

        self.reference_mesh = Mesh(filename=template_file)

        self.mean = np.load(datapath + '/mean.npy')
        self.std = np.load(datapath + '/std.npy')
        self.n_vertex = self.mean.shape[0]
        self.n_features = self.mean.shape[1]