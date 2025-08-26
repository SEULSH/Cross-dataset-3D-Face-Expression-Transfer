
import numpy as np

def CylindricalUnwrap(x):
    cy_coords = np.zeros_like(x)
    depth = np.sqrt(x[:, 0]**2 + x[:, 2]**2)  # sqrt(x**2+z**2)
    theta = np.arctan2(x[:, 0], x[:, 2])  # arctan2(x/z)
    z = x[:, 1]   # y
    cy_coords[:, 0] = theta #
    cy_coords[:, 1] = z  #
    cy_coords[:, 2] = depth #
    return cy_coords


# # Cylindrical unwrapping without annotations
# point = np.load('./add_z_neutral_average_cropped.npy')
# transformedpoint = CylindricalUnwrap(point)
#
# theta = transformedpoint[:, 0]  # for U
# theta_max = max(theta)
# theta_min = min(theta)
# theta_length = theta_max - theta_min
# theta = theta - theta_min  # 0~theta_length
# theta_unit = theta / theta_length  # 0~1
# U = (theta_unit * 126 + 1)  # 1~127 float
#
#
# y = transformedpoint[:, 1]  # for V
# y_max = max(y)
# y_min = min(y)
# y_length = y_max - y_min
# y = y - y_min
# y_unit = y / y_length
# V = (y_unit * 126 + 1)  # 1~127 float
#
# save_uv = np.zeros([len(V), 3])
# save_uv[:,0] = U
# save_uv[:,1] = V
# np.save('SAVE_uv_cropped.npy', save_uv)



