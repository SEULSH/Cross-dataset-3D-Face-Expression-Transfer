##################### test reconstrecution accuracy #########################
import os
import torch
import random
from tqdm import tqdm
import numpy as np
import pickle
import scipy.sparse as sp
import trimesh
from torch.utils.data import DataLoader
from psbody.mesh import Mesh

from utils import get_adj, sparse_mx_to_torch_sparse_tensor
from data import COMA_DATA
from shape_data import ShapeData
from model_for_GT import PaiAutoencoder

from model import Autoencoder

device = torch.device("cuda")
meshpackage = 'mpi-mesh'
dataname = 'COMA_data'
root_dir = '/home/lsh/lsh_data/'
datapath = root_dir + dataname
template_path = datapath + '/template'

UV_path = '/home/lsh/lsh_data/coma_for_expression_transfer_based_UV'

### build autoencoder model and load pretrained parameters for generating GT expression transferred shapes
latent_size = 512
generative_model = 'tiny-conv'
downsample_method = 'COMA_downsample'
reference_mesh_file = os.path.join(template_path, 'template.obj')
downsample_directory = os.path.join(template_path, downsample_method)
ds_factors = [4, 4, 4, 4]
kernal_size = [9, 9, 9, 9, 9]
step_sizes = [2, 2, 1, 1, 1]
filter_sizes_enc = [3, 16, 32, 64, 128]
filter_sizes_dec = [128, 64, 32, 32, 16, 3]
args = {'generative_model': generative_model,
        'datapath': datapath,
        'results_folder': os.path.join(datapath, 'results/' + generative_model),
        'reference_mesh_file': reference_mesh_file, 'downsample_directory': downsample_directory,
        'checkpoint_file': 'checkpoint',
        'seed': 2, 'loss': 'l1',
        'batch_size': 32, 'num_epochs': 300, 'eval_frequency': 200, 'num_workers': 4,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz': latent_size,
        'ds_factors': ds_factors, 'step_sizes': step_sizes,
        'lr': 1e-3, 'regularization': 5e-5,
        'scheduler': True, 'decay_rate': 0.99, 'decay_steps': 1,
        'resume': True,
        'mode': 'test', 'shuffle': True, 'nVal': 100, 'normalization': True}
np.random.seed(args['seed'])
print("Loading data .. ")
shapedata = ShapeData(datapath=args['datapath'],
                       normalization=args['normalization'],
                       template_file=args['reference_mesh_file'],
                       meshpackage=meshpackage)
print("Loading Transform Matrices ..")
with open(os.path.join(args['downsample_directory'], 'downsampling_matrices.pkl'), 'rb') as fp:
    downsampling_matrices = pickle.load(fp)
M_verts_faces = downsampling_matrices['M_verts_faces']
if shapedata.meshpackage == 'mpi-mesh':
    M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
elif shapedata.meshpackage == 'trimesh':
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i
         in range(len(M_verts_faces))]
A = downsampling_matrices['A']
D = downsampling_matrices['D']
U = downsampling_matrices['U']
# add zero last points for each level template
vertices = [torch.cat(
    [torch.tensor(M_verts_faces[i][0], dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)], 0).to(device) for i in range(len(M_verts_faces))]
#
if shapedata.meshpackage == 'mpi-mesh':
    sizes = [x.v.shape[0] for x in M]
elif shapedata.meshpackage == 'trimesh':
    sizes = [x.vertices.shape[0] for x in M]
if not os.path.exists(os.path.join(args['downsample_directory'], 'pai_matrices.pkl')):
    Adj = get_adj(A)
    bU = []
    bD = []
    for i in range(len(D)):  ## add I for last added point
        d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
        u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
        d[0, :-1, :-1] = D[i].todense()
        u[0, :-1, :-1] = U[i].todense()
        d[0, -1, -1] = 1
        u[0, -1, -1] = 1
        bD.append(d)
        bU.append(u)
    bD = [sp.csr_matrix(s[0, ...]) for s in bD]
    bU = [sp.csr_matrix(s[0, ...]) for s in bU]
    with open(os.path.join(args['downsample_directory'], 'pai_matrices.pkl'), 'wb') as fp:
        pickle.dump([Adj, sizes, bD, bU], fp)
else:
    print("Loading adj Matrices ..")
    with open(os.path.join(args['downsample_directory'], 'pai_matrices.pkl'), 'rb') as fp:
        [Adj, sizes, bD, bU] = pickle.load(fp)

tD = [sparse_mx_to_torch_sparse_tensor(s) for s in bD]
tU = [sparse_mx_to_torch_sparse_tensor(s) for s in bU]

model = PaiAutoencoder(filters_enc=args['filter_sizes_enc'],
                       filters_dec=args['filter_sizes_dec'],
                       latent_size=args['nz'],
                       sizes=sizes,
                       t_vertices=vertices,  # template vertex after add last zero nodes
                       num_neighbors=kernal_size,
                       x_neighbors=Adj,
                       D=tD, U=tU).to(device)

checkpoint_path = os.path.join(args['results_folder'], 'latent_'+str(latent_size), 'checkpoints')
checkpoint_dict = torch.load(os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar'),
                             map_location=device)
model_dict = model.state_dict()
pretrained_dict = checkpoint_dict['autoencoder_state_dict']
if next(iter(pretrained_dict)).startswith("module."):
    for k, v in pretrained_dict.items():
        name = k[7:]  # remove `module.`
        model_dict[name] = v
model.load_state_dict(model_dict)
model.to("cpu")
model.eval()

###### build expression transfer network and train the network, build dataset......
save_root = './saved_model' + '/' + dataname
if not os.path.exists(save_root):
    os.makedirs(save_root)

print('training on COMA dataset')
# dataset = COMA_DATA(datapath=datapath, uv_datapath=UV_path, mode='train', model=model, shapedata=shapedata)
dataset = COMA_DATA(datapath=datapath, uv_datapath=UV_path, mode='test', model=model, shapedata=shapedata)

dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=10)

our_model = Autoencoder()
our_model.cuda()
total = sum(p.numel() for p in our_model.parameters())
print("Total params of our model: %.2fM" % (total / 1e6))

checkpoint = torch.load(save_root+'/best.model', map_location=device)
# checkpoint = torch.load(save_root+'/99.model', map_location=device)
# checkpoint = torch.load('./saved_model/COMA_data_0.3333+(1)/99.model', map_location=device)
# checkpoint = torch.load('./saved_model/COMA_data_0.3333+(2)/99.model', map_location=device)
# checkpoint = torch.load('./saved_model/COMA_data_0.3333+(1,2)/99.model', map_location=device)
# checkpoint = torch.load('./saved_model/COMA_data_0.3333+(2,3)/99.model', map_location=device)
# checkpoint = torch.load('./saved_model/COMA_data_replace_decoder/99.model', map_location=device)
our_model.load_state_dict(checkpoint)

best_loss = 1000000.0
for epoch in range(0, 1):

    total_loss = 0
    total_rec_loss = 0
    total_edge_loss = 0
    for j, data in enumerate(tqdm(dataloader)):

        target_shape, source_shape, GT_shape, new_face_T = data
        target_shape = target_shape.to(device)
        source_shape = source_shape.to(device)
        GT_shape = GT_shape.to(device)
        with torch.no_grad():
            deformed_shape = our_model(source_shape.permute(0, 3, 1, 2), target_shape)

        r_loss = torch.mean(torch.sqrt(torch.sum((deformed_shape - GT_shape)**2, dim=-1)))

        loss = 1000.0 * r_loss

        total_loss = total_loss + loss.item()
        total_rec_loss = total_rec_loss + 1000.0 * r_loss.item()

    mean_loss = float((total_loss / (j + 1)))
    mean_rec_loss = float((total_rec_loss / (j + 1)))
    print('epoch: ', epoch, 'mean_loss: ', mean_loss)
    print('epoch: ', epoch, 'mean_rec_loss: ', mean_rec_loss)

print('done')


#################transfer facescape 2 coma #########################
# simplify facescape face to 5023 vertices

import pickle
import os
import numpy as np
import torch
from psbody.mesh import Mesh
from model import Autoencoder

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape
    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

model = Autoencoder()
model.cuda()
model.load_state_dict(torch.load('saved_model/COMA_data_0.3333_200_100+(1,2,3)/best.model', map_location=torch.device(0)))

target_path = '/home/lsh/lsh_data/COMA_data/target_for_transfer_facescape_2_coma/23'
source_path = '/home/lsh/lsh_data/facescape/sources_for_transfer_facescape_2_coma/833_UV'
# target_path = '/home/lsh/lsh_data/COMA_data/target_for_transfer_facescape_2_coma/24'
# source_path = '/home/lsh/lsh_data/facescape/sources_for_transfer_facescape_2_coma/834_UV'

save_dir = './transfer/facescape_2_coma'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for source_file in os.listdir(source_path):

    source_shape = np.load(source_path + '/' + source_file) / 255.0
    source_shape = torch.tensor(source_shape, dtype=torch.float32)
    source_shape = source_shape.cuda()

    for target_file in os.listdir(target_path):
        target_mesh = Mesh(filename=target_path + '/' + target_file)
        target_shape = target_mesh.v
        target_shape = torch.tensor(target_shape, dtype=torch.float32).cuda()
        target_shape = (target_shape - torch.mean(target_shape, dim=0))

        npoints = target_shape.shape[0]
        random_sample2 = np.random.choice(npoints, size=npoints, replace=False)
        target_shape = target_shape[random_sample2]

        with torch.no_grad():
            deformed_shape = model(source_shape.unsqueeze(0).permute(0, 3, 1, 2), target_shape.unsqueeze(0))
        deformed_shape = deformed_shape.squeeze(0)
        deformed_shape = deformed_shape.cpu().detach().numpy()

        temp_array = np.zeros_like(deformed_shape)
        for i in range(npoints):
            index = random_sample2[i]
            temp_array[index] = deformed_shape[i]

        save_mesh = Mesh(v=temp_array, f=target_mesh.f)
        save_name = source_file.replace('.npy', '')+'_2_'+target_file.replace('.ply', '.obj')
        save_mesh.write_obj(save_dir + '/' + save_name)

print('done')


# #############################################################
# ####### test for reconstruction error using selected shapes, not used
# #############################################################
#
# import pickle
# import os
# import numpy as np
# import torch
# from psbody.mesh import Mesh
# from model_5 import NPT
#
# def face_reverse(faces, random_sample):
#     identity_faces=faces
#     face_dict = {}
#     for i in range(len(random_sample)):
#         face_dict[random_sample[i]] = i
#     new_f = []
#     for i in range(len(identity_faces)):
#         new_f.append([face_dict[identity_faces[i][0]], face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
#     new_face = np.array(new_f)
#     return new_face
#
# def scipy_to_torch_sparse(scp_matrix):
#     values = scp_matrix.data
#     indices = np.vstack((scp_matrix.row, scp_matrix.col))
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = scp_matrix.shape
#     sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#     return sparse_tensor
#
# model = NPT()
# model.cuda()
# model.load_state_dict(torch.load('saved_model/COMA_data_0.3333_200_100+(1,2,3)/best.model', map_location=torch.device(0)))
# # NPT_model.eval()
#
# datapath = '/home/lsh/lsh_data/COMA_data/'
# # testdata_path = datapath
#
# template = Mesh(filename=datapath+'template/template.obj')
#
# save_dir = './transfer'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# # source_files = os.listdir(testdata_path+'source_shapes_UV')
# # target_files = os.listdir(testdata_path+'target_shapes')
# # reference_path = '/home/lsh/lsh_data/COMA_data/reference_shapes/target_170915_00223'
#
# # target_files = os.listdir(testdata_path+'source_shapes')
# # source_files = os.listdir(testdata_path+'target_shapes_UV')
# # reference_path = '/home/lsh/lsh_data/COMA_data/reference_shapes/target_170731_00024'
#
#
# source_path = '/home/lsh/lsh_data/COMA_data/shapes_from_train/FaceTalk_170908_03277_TA_UV'
# source_files = os.listdir(source_path)
# target_path = '/home/lsh/lsh_data/COMA_data/shapes_from_train/FaceTalk_170912_03278_TA'
# target_files = os.listdir(target_path)
# reference_path = '/home/lsh/lsh_data/COMA_data/results/tiny-conv/latent_512/predictions/transfer/908_2_912'
#
# # target_path = '/home/lsh/lsh_data/COMA_data/shapes_from_train/FaceTalk_170908_03277_TA'
# # target_files = os.listdir(target_path)
# # source_path = '/home/lsh/lsh_data/COMA_data/shapes_from_train/FaceTalk_170912_03278_TA_UV'
# # source_files = os.listdir(source_path)
# # reference_path = '/home/lsh/lsh_data/COMA_data/results/tiny-conv/latent_512/predictions/transfer/912_2_908'
#
# count = 0
# error_sum = 0
#
# for source_file in source_files:
#
#     # source_shape = np.load(testdata_path + 'source_shapes_UV' + '/' + source_file) / 255.0
#     # source_shape = np.load(testdata_path + 'target_shapes_UV' + '/' + source_file) / 255.0
#
#     source_shape = np.load(source_path + '/' + source_file) / 255.0
#
#     source_shape = torch.tensor(source_shape, dtype=torch.float32)
#     source_shape = source_shape.cuda()
#
#     reference_shape = Mesh(filename=reference_path+'/'+source_file.split('.')[0]+'_t.obj').v
#     reference_shape = reference_shape - np.mean(reference_shape, axis=0)
#
#     for target_file in target_files:
#         # target_mesh = Mesh(filename=testdata_path + 'target_shapes' + '/' + target_file)
#         # target_mesh = Mesh(filename=testdata_path + 'source_shapes' + '/' + target_file)
#
#         target_mesh = Mesh(filename=target_path + '/' + target_file)
#
#         target_shape = target_mesh.v
#         target_shape = torch.tensor(target_shape, dtype=torch.float32).cuda()
#         target_shape = (target_shape - torch.mean(target_shape, dim=0))
#
#         npoints = target_shape.shape[0]
#         random_sample2 = np.random.choice(npoints, size=npoints, replace=False)
#         target_shape = target_shape[random_sample2]
#
#         with torch.no_grad():
#             deformed_shape = model(source_shape.unsqueeze(0).permute(0, 3, 1, 2), target_shape.unsqueeze(0))
#
#         deformed_shape = deformed_shape.squeeze(0)
#         deformed_shape = deformed_shape.cpu().detach().numpy()
#
#         temp_array = np.zeros_like(deformed_shape)
#         for i in range(npoints):
#             index = random_sample2[i]
#             temp_array[index] = deformed_shape[i]
#
#         error = np.mean(np.sqrt(np.sum((reference_shape - temp_array)**2, axis=1)))
#         error_sum = error_sum + error
#         count = count + 1
#
#         save_mesh = Mesh(v=temp_array, f=template.f)
#         save_name = source_file.split('.')[0]+'_2_'+target_file.split('.')[0]+'.obj'
#         save_mesh.write_obj(save_dir + '/' + save_name)
#
# print('mean error: ', error_sum / count)
#
# print('')
# # 0.0005102394626183233 + 0.0005432789060588887
# ##########################################################################





