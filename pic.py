import matplotlib
import numpy as np
import os
from setting import out_pic_dic,is_out_pic_example
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
ap0 = [[-0.00001, -0.00002, 1.00830],
       [0.14475, -0.07537, 0.99674],
       [0.13747, -0.10477, 0.51222],
       [0.05090, -0.27371, 0.07246],
       [-0.14476, 0.07535, 1.01964],
       [-0.19476, -0.03135, 0.54859],
       [-0.33582, -0.27418, 0.14530],
       [0.03465, 0.07187, 1.27154],
       [0.08203, 0.09819, 1.55611],
       [0.13682, 0.16097, 1.71822],
       [0.09326, 0.02394, 1.77841],
       [-0.10811, 0.13847, 1.54135],
       [-0.40501, 0.27080, 1.54436],
       [-0.48869, 0.47729, 1.79511],
       [0.21069, -0.01737, 1.48483],
       [0.39367, -0.21920, 1.31686],
       [0.63089, -0.16669, 1.51139]]
def pic(ap0):
    ap = np.array(ap0, dtype='float32')
    np_data = ap
    xp = np_data.T[0].T
    yp = np_data.T[1].T
    zp = np_data.T[2].T

    ax = plt.axes(projection='3d')

    radius = 1.7
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.view_init(elev=15., azim=70)
    ax.dist = 7.5

    # 3D scatter
    ax.scatter3D(xp, yp, zp, cmap='Greens')

    # left leg, node [0, 1, 2, 3]
    ax.plot(xp[0:4], yp[0:4], zp[0:4], ls='-', color='red')

    # right leg
    ax.plot(np.hstack((xp[0], xp[4:7])),
            np.hstack((yp[0], yp[4:7])),
            np.hstack((zp[0], zp[4:7])),
            ls='-', color='blue')

    # spine, node [0, 7, 8, 9, 10]
    ax.plot(np.hstack((xp[0], xp[7:11])),
            np.hstack((yp[0], yp[7:11])),
            np.hstack((zp[0], zp[7:11])),
            ls='-', color='gray')

    # right arm, node [8, 11, 12, 13]
    ax.plot(np.hstack((xp[8], xp[11:14])),
            np.hstack((yp[8], yp[11:14])),
            np.hstack((zp[8], zp[11:14])),
            ls='-', color='blue')

    # left arm, node [8, 14, 15, 16]
    ax.plot(np.hstack((xp[8], xp[14:])),
            np.hstack((yp[8], yp[14:])),
            np.hstack((zp[8], zp[14:])),
            ls='-', color='red')
    data=str(datetime.datetime.now().strftime("%Y-%m-%d %H %M"))
    if out_pic_dic:
        if (os.path.exists(out_pic_dic+"/"+data) == False):
            os.makedirs(out_pic_dic+"/"+data)
            plt.savefig(out_pic_dic+"/"+data+"/"+str(datetime.datetime.now().strftime("%S"))+".jpg")


if __name__ == '__main__':
    if is_out_pic_example:
        print("事例代码")
        pic(ap0)
