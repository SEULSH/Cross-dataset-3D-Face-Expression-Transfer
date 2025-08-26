import os
import torch.utils.data as data
import torch
import numpy as np
import math
import random
import pickle
from psbody.mesh import Mesh
from skimage.io import imread
import torch

class COMA_DATA(data.Dataset):
    def __init__(self, datapath=None, uv_datapath=None, mode='train', model=None, shapedata=None):

        self.path_files = []
        self.persons = []
        self.datapath = datapath + '/' + mode
        for p in os.listdir(self.datapath):
            self.persons.append(p)
            for ex in os.listdir(self.datapath+'/'+p):
                for face in os.listdir(self.datapath+'/'+p+'/'+ex):
                    self.path_files.append(self.datapath+'/'+p+'/'+ex+'/'+face)

        self.neutral_path = datapath + '/neutral_faces'

        self.uv_datapath = uv_datapath + '/' + mode

        self.model = model
        self.shapedata = shapedata
        self.mean = torch.tensor(self.shapedata.mean, dtype=torch.float32)
        self.std = torch.tensor(self.shapedata.std, dtype=torch.float32)
        self.template = shapedata.reference_mesh

    def __getitem__(self, index):

        source_E_shape = Mesh(filename=self.path_files[index]).v
        source_E_shape = torch.tensor(source_E_shape, dtype=torch.float32)
        source_E_identity = self.path_files[index].split('/')[-3]
        source_N_shape = Mesh(filename=self.neutral_path+'/'+source_E_identity+'.ply').v
        source_N_shape = torch.tensor(source_N_shape, dtype=torch.float32)
        source_E_shape = (source_E_shape - self.mean) / self.std
        source_N_shape = (source_N_shape - self.mean) / self.std

        target_id = np.random.randint(0, len(self.path_files))
        target_E_shape = Mesh(filename=self.path_files[target_id]).v
        target_E_shape = torch.tensor(target_E_shape, dtype=torch.float32)

        target_E_identity = self.path_files[target_id].split('/')[-3]
        target_N_shape = Mesh(filename=self.neutral_path+'/'+target_E_identity+'.ply').v
        target_N_shape = torch.tensor(target_N_shape, dtype=torch.float32)
        target_N_shape = (target_N_shape - self.mean) / self.std

        source_paths = self.path_files[index].split('/')
        source_image = torch.tensor(np.load(self.uv_datapath+'/'+source_paths[-3]+'/'+source_paths[-2]+'/'+source_paths[-1].replace('.ply', '.npy')) / 255, dtype=torch.float32)

        with torch.no_grad():
            generated_GT_shape = self.model(source_N_shape, source_E_shape, target_N_shape)[0:-1]

        generated_GT_shape = generated_GT_shape * self.std + self.mean

        # # # # ## normalize
        target_E_shape = target_E_shape - torch.mean(target_E_shape, dim=0)
        generated_GT_shape = generated_GT_shape - torch.mean(generated_GT_shape, dim=0)

        npoints = generated_GT_shape.shape[0]
        random_sample = np.random.choice(npoints, size=npoints, replace=False)
        target_E_shape = target_E_shape[random_sample]
        generated_GT_shape = generated_GT_shape[random_sample]

        new_face_T = []
        return target_E_shape, source_image, generated_GT_shape, new_face_T

    def __len__(self):
        return len(self.path_files)