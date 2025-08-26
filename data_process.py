import numpy as np
from psbody.mesh import Mesh
import os
from unwrap_code import CylindricalUnwrap
from skimage.io import imread, imsave
from face3d import mesh
from face3d.mesh.render import render_colors

#################################################
# ## allign COMA shape to mean face of facescape
# from ICP import icp
# COMA_path = '/home/lsh/lsh_data/COMA_data/shapes_from_train/FaceTalk_170904_03276_TA'
# average_shape = Mesh(filename='neutral_average_save.obj').v
# save_path = 'aligned_coma_faces'
# if os.path.exists(save_path)==False:
#     os.makedirs(save_path)
#
# COMA_template = Mesh(filename='/home/lsh/lsh_data/COMA_data/template/template.obj')
# template_shape = COMA_template.v * 1000.0
# New_template = Mesh(v=template_shape, f=COMA_template.f)
# New_template.write_obj(save_path+'/'+'new_COMA_template.obj')
#
# for file in os.listdir(COMA_path):
#     COMA_face = Mesh(filename=COMA_path+'/'+file)
#     COMA_shape = COMA_face.v * 1000.0
#
#     New_COMA_face = Mesh(v=COMA_shape, f=COMA_face.f)
#     New_COMA_face.write_obj(save_path+'/'+file.replace('ply', 'obj'))
#
#     # T, _, i = icp(COMA_shape, average_shape, max_iterations=10000, tolerance=0.00000000001)
#     # C = np.ones((COMA_shape.shape[0], 4))
#     # C[:, 0:3] = np.copy(COMA_shape)
#     # C = np.dot(T, C.T).T
#     # transform_shape = C[:, 0:3]
#     # save_mesh = Mesh(v=transform_shape, f=COMA_face.f)
#     # save_mesh.write_obj(save_path+'/'+file.replace('ply', 'obj'))
#     print(file)
#
# print('')

####################################################

# # # process two bad face shapes
# # from ICP import icp
# # mesh0 = Mesh(filename='neutral_average_save.obj')
# # f = mesh0.f
# # face0 = mesh0.v
# # face1 = np.load('/home/lsh/lsh_data/facescape/719_jaw_left.npy')
# # mesh1 = Mesh(v=face1, f=f)
# # mesh1.write_obj('719_jaw_left.obj')
# # face2 = np.load('/home/lsh/lsh_data/facescape/721_jaw_left.npy')
# # mesh2 = Mesh(v=face2, f=f)
# # mesh2.write_obj('721_jaw_left.obj')
# #
# # T, _, i = icp(face1, face0, max_iterations=10000, tolerance=0.00000000001)
# # C = np.ones((face1.shape[0], 4))
# # C[:, 0:3] = np.copy(face1)
# # C = np.dot(T, C.T).T
# # transform_shape = C[:, 0:3]
# # np.save('T_719_jaw_left.npy', transform_shape)
# # mesh11 = Mesh(v=transform_shape, f=f)
# # mesh11.write_obj('T_719_jaw_left.obj')
# #
# # T, _, i = icp(face2, face0, max_iterations=10000, tolerance=0.00000000001)
# # C = np.ones((face2.shape[0], 4))
# # C[:, 0:3] = np.copy(face2)
# # C = np.dot(T, C.T).T
# # transform_shape = C[:, 0:3]
# # np.save('T_721_jaw_left.npy', transform_shape)
# # mesh22 = Mesh(v=transform_shape, f=f)
# # mesh22.write_obj('T_721_jaw_left.obj')
# # print('')
# face_719 = Mesh(filename='/home/lsh/lsh_data/facescape/719_jaw_left.obj').v
# np.save('/home/lsh/lsh_data/facescape/719_jaw_left.npy', face_719)
# face_721 = Mesh(filename='/home/lsh/lsh_data/facescape/721_jaw_left.obj').v
# np.save('/home/lsh/lsh_data/facescape/721_jaw_left.npy', face_721)
# print('')

# bad_face = np.load('/home/lsh/lsh_data/facescape/65_lip_roll.npy')
# mesh0 = Mesh(filename='neutral_average_save.obj')
# show_face = Mesh(v=bad_face, f=mesh0.f)
# show_face.write_obj('/home/lsh/lsh_data/facescape/65_lip_roll.obj')

# shape = Mesh(filename='/home/lsh/lsh_data/facescape/65_lip_roll.obj').v
# np.save('/home/lsh/lsh_data/facescape/65_lip_roll.npy', shape)
# print('')
###############################################################

# template_path = '/home/lsh/lsh_data/facescape/template/template.obj'
# face = Mesh(filename=template_path).f
#
# # file_path = '/home/lsh/lsh_data/facescape/eval'
# # save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/eval'
# # file_path = '/home/lsh/lsh_data/facescape/test'
# # save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/test'
# file_path = '/home/lsh/lsh_data/facescape/train'
# save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/train'
# if os.path.exists(save_path)==False:
#     os.makedirs(save_path)
#
# for file in os.listdir(file_path):
#     vertices = np.load(file_path+'/'+file)
#     save_mesh = Mesh(v=vertices, f=face)
#     save_name = file.split('.')[0]+'.obj'
#     save_mesh.write_obj(save_path+'/'+save_name)

#########################################################

# neutral_mesh = Mesh(filename='neutral_average_save.obj')
# neutral_vertices = neutral_mesh.v
# cropped_mesh = Mesh(filename='neutral_average_cropped.obj')
# cropped_vertices = cropped_mesh.v
# index = []
# for i in range(cropped_vertices.shape[0]):
#     vi = cropped_vertices[i]
#     for j in range(neutral_vertices.shape[0]):
#         vj = neutral_vertices[j]
#         if np.array_equal(vi, vj):
#             index.append(j)
#             break
# index = np.array(index, dtype=int)
# np.save('index.npy', index)
# print('')

# ## for COMA data
# neutral_mesh = Mesh(filename='save_bareteeth.000100.obj')
# neutral_vertices = neutral_mesh.v
# cropped_mesh = Mesh(filename='new_bareteeth.000100.obj')
# cropped_vertices = cropped_mesh.v
# index = []
# for i in range(cropped_vertices.shape[0]):
#     vi = cropped_vertices[i]
#     for j in range(neutral_vertices.shape[0]):
#         vj = neutral_vertices[j]
#         if np.array_equal(vi, vj):
#             index.append(j)
#             break
# index = np.array(index, dtype=int)
# np.save('coma_cropped_index.npy', index)
# print('')


################################################

# cropped_mesh = Mesh(filename='neutral_average_cropped.obj')
# face = cropped_mesh.f
# index = np.load('index.npy')
# test_shape = np.load('/home/lsh/lsh_data/facescape/eval/801_anger.npy')
# new_vertices = test_shape[index]
# test_mesh = Mesh(v=new_vertices, f=face)
# test_mesh.write_obj('test_mesh.obj')

# #####################for facescape uv images##############################
#
# def process_uv(uv_coords, uv_h = 256, uv_w = 256):
#     uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)  # 0~255
#     uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)  # 0~255
#     uv_coords[:,1] = uv_h - uv_coords[:,1] - 1  # 0~255  # inverse V values
#     uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z = 0,0,0,...
#     return uv_coords
#
# cropped_mesh = Mesh(filename='neutral_average_cropped.obj')
# my_faces = cropped_mesh.f
# index = np.load('index.npy')
#
# image_w = image_h = 128
#
# # file_path = '/home/lsh/lsh_data/facescape/train'
# # save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/train'
# file_path = '/home/lsh/lsh_data/facescape/test'
# save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/test'
# # file_path = '/home/lsh/lsh_data/facescape/eval'
# # save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/eval'
# if os.path.exists(save_path)==False:
#     os.makedirs(save_path)
#
# # Z_min = 100000      # train-82.834814, test-71.840937, eval-65.799784,
# # X_len_max = 0  # train215.121065, test194.57226400000002, eval189.870424,
# # Y_len_max = 0  # train289.957914, test281.17663600000003, eval275.163235,
# # Z_len_max = 0  # train155.05469699999998, test144.401185, eval143.778891,
#
# for file in os.listdir(file_path):
#     ori_vertices = np.load(file_path+'/'+file)
#     vertices = ori_vertices[index]
#
#     # if min(vertices[:, 2]) < Z_min:
#     #     Z_min = min(vertices[:, 2])
#     #     file_name0 = file
#
#     vertices[:, 2] = vertices[:, 2] + 83.0
#
#     transformedpoint = CylindricalUnwrap(vertices)
#     theta = transformedpoint[:, 0]  # for U
#     theta_max = max(theta)
#     theta_min = min(theta)
#     theta_length = theta_max - theta_min
#     theta = theta - theta_min  # 0~theta_length
#     U = theta / theta_length  # 0~1
#     y = transformedpoint[:, 1]  # for V
#     y_max = max(y)
#     y_min = min(y)
#     y_length = y_max - y_min
#     y = y - y_min
#     V = y / y_length  # 0~1
#     UV_coords = np.zeros([U.shape[0], 2])
#     UV_coords[:, 0] = U
#     UV_coords[:, 1] = V
#     UV_coords = process_uv(UV_coords, image_w, image_h)
#
#     my_points = vertices
#
#     # X_len = (max(my_points[:, 0]) - min(my_points[:, 0]))
#     # if X_len > X_len_max:
#     #     X_len_max = X_len
#     #     file_name1 = file
#     my_points[:, 0] = (my_points[:, 0] - min(my_points[:, 0])) / 216 * 255
#     # Y_len = (max(my_points[:, 1]) - min(my_points[:, 1]))
#     # if Y_len > Y_len_max:
#     #     Y_len_max = Y_len
#     #     file_name2 = file
#     my_points[:, 1] = (my_points[:, 1] - min(my_points[:, 1])) / 290 * 255
#     # Z_len = (max(my_points[:, 2]) - min(my_points[:, 2]))
#     # if Z_len > Z_len_max:
#     #     Z_len_max = Z_len
#     #     file_name3 = file
#     my_points[:, 2] = (my_points[:, 2] - min(my_points[:, 2])) / 156 * 255
#
#     uv_position_map = mesh.render.render_colors(UV_coords, my_faces, my_points, image_w, image_h, c=3)
#
#     uv_position_map = uv_position_map[17:117, 14:114, :]  # up 2 down, left 2 right
#
#     # imsave('{}/{}'.format(save_path, file.replace('.npy', '.jpg')), np.uint8(uv_position_map))
#
#     np.save(save_path+'/'+file, uv_position_map)
#
#     # print('')
#
# # print('z_min: ', Z_min)
# # print('X_len_max: ', X_len_max)
# # print('Y_len_max: ', Y_len_max)
# # print('Z_len_max: ', Z_len_max)
# # print(file_name0)
# # print(file_name1)
# # print(file_name2)
# # print(file_name3)

# ####################### get a UV example of facescape ###############################################
# def process_uv(uv_coords, uv_h = 256, uv_w = 256):
#     uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)  # 0~255
#     uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)  # 0~255
#     uv_coords[:,1] = uv_h - uv_coords[:,1] - 1  # 0~255  # inverse V values
#     uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z = 0,0,0,...
#     return uv_coords
#
# cropped_mesh = Mesh(filename='neutral_average_cropped.obj')
# my_faces = cropped_mesh.f
# index = np.load('index.npy')
#
# image_w = image_h = 128
#
# file_path = '/home/lsh/lsh_data/facescape/eval_example'
# save_path = '/home/lsh/lsh_data/facescape_for_expression_transfer_based_UV/eval_example'
# if os.path.exists(save_path)==False:
#     os.makedirs(save_path)
#
# for file in os.listdir(file_path):
#     ori_vertices = np.load(file_path+'/'+file)
#     vertices = ori_vertices[index]
#
#     d_mesh = Mesh(v=vertices, f=my_faces)
#     d_mesh.write_obj(save_path+'/'+file.replace('npy', 'obj'))
#
#     vertices[:, 2] = vertices[:, 2] + 83.0
#
#     transformedpoint = CylindricalUnwrap(vertices)
#     theta = transformedpoint[:, 0]  # for U
#     theta_max = max(theta)
#     theta_min = min(theta)
#     theta_length = theta_max - theta_min
#     theta = theta - theta_min  # 0~theta_length
#     U = theta / theta_length  # 0~1
#     y = transformedpoint[:, 1]  # for V
#     y_max = max(y)
#     y_min = min(y)
#     y_length = y_max - y_min
#     y = y - y_min
#     V = y / y_length  # 0~1
#     UV_coords = np.zeros([U.shape[0], 2])
#     UV_coords[:, 0] = U
#     UV_coords[:, 1] = V
#     UV_coords = process_uv(UV_coords, image_w, image_h)
#
#     my_points = vertices
#
#     my_points[:, 0] = (my_points[:, 0] - min(my_points[:, 0])) / 216 * 255
#
#     my_points[:, 1] = (my_points[:, 1] - min(my_points[:, 1])) / 290 * 255
#
#     my_points[:, 2] = (my_points[:, 2] - min(my_points[:, 2])) / 156 * 255
#
#     uv_position_map = mesh.render.render_colors(UV_coords, my_faces, my_points, image_w, image_h, c=3)
#
#     # uv_position_map = uv_position_map[17:117, 14:114, :]  # up 2 down, left 2 right
#
#     imsave('{}/{}'.format(save_path, file.replace('.npy', '.jpg')), np.uint8(uv_position_map))


# ####################for COMA UV images##################################################

def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)  # 0~255
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)  # 0~255
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1  # 0~255  # inverse V values
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z = 0,0,0,...
    return uv_coords

cropped_mesh = Mesh(filename='new_bareteeth.000100.obj')
my_faces = cropped_mesh.f
index = np.load('coma_cropped_index.npy')

# image_w = image_h = 128
image_w = image_h = 146

# file_path = '/home/lsh/lsh_data/COMA_data/train'
# # z_min:  -60.06859615445137
# # X_len_max:  161.13535314798355
# # Y_len_max:  251.30073726177216
# # Z_len_max:  136.19456440210342
# # /home/lsh/lsh_data/COMA_data/train/FaceTalk_170904_03276_TA/mouth_extreme/mouth_extreme.000039.ply
# # /home/lsh/lsh_data/COMA_data/train/FaceTalk_170728_03272_TA/mouth_side/mouth_side.000133.ply
# # /home/lsh/lsh_data/COMA_data/train/FaceTalk_170809_00138_TA/mouth_extreme/mouth_extreme.000076.ply
# # /home/lsh/lsh_data/COMA_data/train/FaceTalk_170725_00137_TA/cheeks_in/cheeks_in.000059.ply

file_path = '/home/lsh/lsh_data/COMA_data/test'
# # z_min:  -59.54933911561966
# # X_len_max:  158.62146764993668
# # Y_len_max:  249.07968193292618
# # Z_len_max:  133.51867720484734
# # /home/lsh/lsh_data/COMA_data/test/FaceTalk_170731_00024_TA/cheeks_in/cheeks_in.000100.ply
# # /home/lsh/lsh_data/COMA_data/test/FaceTalk_170915_00223_TA/mouth_extreme/mouth_extreme.000029.ply
# # /home/lsh/lsh_data/COMA_data/test/FaceTalk_170731_00024_TA/mouth_side/mouth_side.000051.ply
# # /home/lsh/lsh_data/COMA_data/test/FaceTalk_170731_00024_TA/cheeks_in/cheeks_in.000118.ply

save_dir = '/home/lsh/lsh_data/coma_for_expression_transfer_based_UV/' + file_path.split('/')[-1]

# Z_min = 100000 #
# X_len_max = 0  #
# Y_len_max = 0  #
# Z_len_max = 0  #

for p in os.listdir(file_path):
    for ex in os.listdir(file_path+'/'+p):

        save_path = save_dir + '/' + p+'/'+ex
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)

        for shape in os.listdir(file_path+'/'+p+'/'+ex):
            shape_path = file_path+'/'+p+'/'+ex + '/' + shape

            ori_mesh = Mesh(filename=shape_path)
            ori_vertices = ori_mesh.v * 1000.0
            vertices = ori_vertices[index]

            # if min(vertices[:, 2]) < Z_min:
            #     Z_min = min(vertices[:, 2])
            #     file_name0 = shape_path

            vertices[:, 2] = vertices[:, 2] + 61.0
            transformedpoint = CylindricalUnwrap(vertices)
            theta = transformedpoint[:, 0]  # for U
            theta_max = max(theta)
            theta_min = min(theta)
            theta_length = theta_max - theta_min
            theta = theta - theta_min  # 0~theta_length
            U = theta / theta_length  # 0~1
            y = transformedpoint[:, 1]  # for V
            y_max = max(y)
            y_min = min(y)
            y_length = y_max - y_min
            y = y - y_min
            V = y / y_length  # 0~1
            UV_coords = np.zeros([U.shape[0], 2])
            UV_coords[:, 0] = U
            UV_coords[:, 1] = V
            UV_coords = process_uv(UV_coords, image_w, image_h)
            my_points = vertices

            # X_len = (max(vertices[:, 0]) - min(vertices[:, 0]))
            # if X_len > X_len_max:
            #     X_len_max = X_len
            #     file_name1 = shape_path

            my_points[:, 0] = (my_points[:, 0] - min(my_points[:, 0])) / 162 * 255

            # Y_len = (max(vertices[:, 1]) - min(vertices[:, 1]))
            # if Y_len > Y_len_max:
            #     Y_len_max = Y_len
            #     file_name2 = shape_path

            my_points[:, 1] = (my_points[:, 1] - min(my_points[:, 1])) / 252 * 255

            # Z_len = (max(vertices[:, 2]) - min(vertices[:, 2]))
            # if Z_len > Z_len_max:
            #     Z_len_max = Z_len
            #     file_name3 = shape_path

            my_points[:, 2] = (my_points[:, 2] - min(my_points[:, 2])) / 137 * 255

            uv_position_map = mesh.render.render_colors(UV_coords, my_faces, my_points, image_w, image_h, c = 3)

            # uv_position_map = uv_position_map[13:113, 14:114, :]  # up 2 down, left 2 right
            uv_position_map = uv_position_map[22:122, 24:124, :]  # up 2 down, left 2 right

            # imsave('{}/{}'.format(save_path, shape.replace('.ply', '.jpg')), np.uint8((uv_position_map)))
            # imsave('{}/{}'.format(save_path, shape.replace('.ply', '.jpg')), uv_position_map) ## wrong code

            np.save(save_path + '/' + shape.replace('ply', 'npy'), uv_position_map)

            # print('')

# print('z_min: ', Z_min)
# print('X_len_max: ', X_len_max)
# print('Y_len_max: ', Y_len_max)
# print('Z_len_max: ', Z_len_max)
# print(file_name0)
# print(file_name1)
# print(file_name2)
# print(file_name3)

# ##################for get a example UV of COMA############################
# def process_uv(uv_coords, uv_h=256, uv_w=256):
#     uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)  # 0~255
#     uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)  # 0~255
#     uv_coords[:,1] = uv_h - uv_coords[:,1] - 1  # 0~255  # inverse V values
#     uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z = 0,0,0,...
#     return uv_coords
#
# cropped_mesh = Mesh(filename='new_bareteeth.000100.obj')
# my_faces = cropped_mesh.f
# index = np.load('coma_cropped_index.npy')
#
# # image_w = image_h = 128
# image_w = image_h = 146
#
# file_path = '/home/lsh/lsh_data/COMA_data/source_shapes'
#
# save_dir = '/home/lsh/lsh_data/coma_for_expression_transfer_based_UV/' + file_path.split('/')[-1]
#
# if os.path.exists(save_dir) == False:
#     os.makedirs(save_dir)
#
# for file in os.listdir(file_path):
#
#         shape_path = file_path+'/'+file
#
#         ori_mesh = Mesh(filename=shape_path)
#         ori_vertices = ori_mesh.v * 1000.0
#         vertices = ori_vertices[index]
#
#         d_mesh = Mesh(v=vertices, f=my_faces)
#         d_mesh.write_obj(save_dir+'/'+file.replace('ply', 'obj'))
#         vertices[:, 2] = vertices[:, 2] + 61.0
#
#         transformedpoint = CylindricalUnwrap(vertices)
#         theta = transformedpoint[:, 0]  # for U
#         theta_max = max(theta)
#         theta_min = min(theta)
#         theta_length = theta_max - theta_min
#         theta = theta - theta_min  # 0~theta_length
#         U = theta / theta_length  # 0~1
#         y = transformedpoint[:, 1]  # for V
#         y_max = max(y)
#         y_min = min(y)
#         y_length = y_max - y_min
#         y = y - y_min
#         V = y / y_length  # 0~1
#         UV_coords = np.zeros([U.shape[0], 2])
#         UV_coords[:, 0] = U
#         UV_coords[:, 1] = V
#         UV_coords = process_uv(UV_coords, image_w, image_h)
#         my_points = vertices
#
#         my_points[:, 0] = (my_points[:, 0] - min(my_points[:, 0])) / 162 * 255
#
#         my_points[:, 1] = (my_points[:, 1] - min(my_points[:, 1])) / 252 * 255
#
#         my_points[:, 2] = (my_points[:, 2] - min(my_points[:, 2])) / 137 * 255
#
#         uv_position_map = mesh.render.render_colors(UV_coords, my_faces, my_points, image_w, image_h, c = 3)
#
#         # uv_position_map = uv_position_map[13:113, 14:114, :]  # up 2 down, left 2 right
#         uv_position_map = uv_position_map[22:122, 24:124, :]  # up 2 down, left 2 right
#
#         imsave('{}/{}'.format(save_dir, file.replace('.ply', '.jpg')), np.uint8((uv_position_map)))


# # ####################for FWH UV images######################################
#
# def process_uv(uv_coords, uv_h=256, uv_w=256):
#     uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)  # 0~255
#     uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)  # 0~255
#     uv_coords[:,1] = uv_h - uv_coords[:,1] - 1  # 0~255  # inverse V values
#     uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z = 0,0,0,...
#     return uv_coords
#
# cropped_mesh = Mesh(filename='/home/lsh/lsh_data/test_used_source_shapes_FWH/data/d_pose_1_cropped.obj')
# my_faces = cropped_mesh.f
# index = np.load('/home/lsh/lsh_data/test_used_source_shapes_FWH/data/d_cropped_FWH_index.npy')
#
# image_w = image_h = 128
#
# file_path = '/home/lsh/lsh_data/test_used_source_shapes_FWH/data/d_data'
# # z_min:  -19.5704
# # X_len_max:  146.29680000000002
# # Y_len_max:  201.84519999999998
# # Z_len_max:  85.59630000000001
# # d_pose_11.obj
# # d_pose_11.obj
# # d_pose_5.obj
# # d_pose_11.obj
#
# save_dir = '/home/lsh/lsh_data/test_used_source_shapes_FWH/data/cropped_uv_data'
#
# # Z_min = 100000 #
# # X_len_max = 0  #
# # Y_len_max = 0  #
# # Z_len_max = 0  #
#
# for file in os.listdir(file_path):
#
#     ori_mesh = Mesh(filename=file_path+'/'+file)
#     ori_vertices = ori_mesh.v * 100.0
#     vertices = ori_vertices[index]
#
#     # if min(vertices[:, 2]) < Z_min:
#     #     Z_min = min(vertices[:, 2])
#     #     file_name0 = file
#
#     vertices[:, 2] = vertices[:, 2] + 20.0
#     transformedpoint = CylindricalUnwrap(vertices)
#     theta = transformedpoint[:, 0]  # for U
#     theta_max = max(theta)
#     theta_min = min(theta)
#     theta_length = theta_max - theta_min
#     theta = theta - theta_min  # 0~theta_length
#     U = theta / theta_length  # 0~1
#     y = transformedpoint[:, 1]  # for V
#     y_max = max(y)
#     y_min = min(y)
#     y_length = y_max - y_min
#     y = y - y_min
#     V = y / y_length  # 0~1
#     UV_coords = np.zeros([U.shape[0], 2])
#     UV_coords[:, 0] = U
#     UV_coords[:, 1] = V
#     UV_coords = process_uv(UV_coords, image_w, image_h)
#     my_points = vertices
#
#     # X_len = (max(vertices[:, 0]) - min(vertices[:, 0]))
#     # if X_len > X_len_max:
#     #     X_len_max = X_len
#     #     file_name1 = file
#
#     my_points[:, 0] = (my_points[:, 0] - min(my_points[:, 0])) / 147 * 255
#
#     # Y_len = (max(vertices[:, 1]) - min(vertices[:, 1]))
#     # if Y_len > Y_len_max:
#     #     Y_len_max = Y_len
#     #     file_name2 = file
#
#     my_points[:, 1] = (my_points[:, 1] - min(my_points[:, 1])) / 202 * 255
#
#     # Z_len = (max(vertices[:, 2]) - min(vertices[:, 2]))
#     # if Z_len > Z_len_max:
#     #     Z_len_max = Z_len
#     #     file_name3 = file
#
#     my_points[:, 2] = (my_points[:, 2] - min(my_points[:, 2])) / 86 * 255
#
#     uv_position_map = mesh.render.render_colors(UV_coords, my_faces, my_points, image_w, image_h, c = 3)
#
#     uv_position_map = uv_position_map[13:113, 14:114, :]  # up 2 down, left 2 right
#
#     # imsave('{}/{}'.format(save_dir, file.replace('.obj', '.jpg')), np.uint8((uv_position_map)))
#
#     np.save(save_dir + '/' + file.replace('obj', 'npy'), uv_position_map)
#
#     # print('')
#
# # print('z_min: ', Z_min)
# # print('X_len_max: ', X_len_max)
# # print('Y_len_max: ', Y_len_max)
# # print('Z_len_max: ', Z_len_max)
# # print(file_name0)
# # print(file_name1)
# # print(file_name2)
# # print(file_name3)
