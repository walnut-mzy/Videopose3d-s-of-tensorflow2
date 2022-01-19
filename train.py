import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
    logical_devices = tf.config.list_logical_devices("CPU")
from model import model
import numpy as np
import hashlib
from tensorflow.keras import losses, optimizers,initializers,metrics
from setting import modelpath,subjects_train,subjects_test,architecture,data_2d_path,data_3d_path,output,loss_l,batch_size,epoch
from amsgrad import AMSGrad
from loss import MPJPE,p_mpjpe
from process.h36m_dataset import Human36mDataset
from process.camera import world_to_camera,normalize_screen_coordinates
from data import data_process, deterministic_random,data_process1
import datetime
import os
from tqdm import tqdm
print(tf.executing_eagerly())
#optimizer=AMSGrad(learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
loss=MPJPE
if loss_l=="p_mpjpe":
    loss=p_mpjpe

acc_meter=tf.metrics.CategoricalAccuracy()
@tf.function
def train_one_step(x,y):
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        predictions = model(x)
        acc_meter.update_state(y_true=y, y_pred=predictions)
        loss1 = loss(y,predictions)
    # 求梯度
    grad = tape.gradient(loss1,  model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    return loss1

# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus)
# if gpus:
#     gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
#     tf.config.set_visible_devices([gpu0], "GPU")
#     tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
#     # 或者也可以设置GPU显存为固定使用量(例如：4G)
#     #tf.config.experimental.set_virtual_device_configuration(gpu0,
#     #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

# @tf.function
# def test_acc():
#     for x1, y1 in test_dataset:
#         pred = model(x1)  # 前向计算
#         acc_meter1.update_state(y_true=y1, y_pred=pred)  # 更新准确率统计
#     print("测试集正确率为：",acc_meter1.result())
#     acc_meter1.reset_states()
if __name__ == '__main__':

    model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
    )
    print("load model")
    dataset = Human36mDataset(data_2d_path)
    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d=pos_3d.numpy()
                    # print(pos_3d[:, :1])
                    # print( pos_3d[:, 1:])
                    aa=tf.tile(pos_3d[:, :1],multiples=[1,16,1])
                    # print(aa)
                    pos_3d[:, 1:] -=aa  # Remove global offset, but keep trajectory in first position
                    positions_3d.append(tf.convert_to_tensor(pos_3d,dtype=float))
                anim['positions_3d'] = positions_3d
    print('Loading 2D detections...')
    keypoints = np.load(data_3d_path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in dataset[subject].keys():
            assert action in keypoints[
                subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = subjects_train.split(',')
    subjects_test = subjects_test.split(',')


    def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        for subject in subjects:
            for action in keypoints[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)):  # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):  # Iterate across cameras
                        out_poses_3d.append(poses_3d[i])

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = 1
        if subset < 1:
            for i in range(len(out_poses_2d)):
                n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d


    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, None)
    filter_widths = [int(x) for x in architecture.split(',')]
    # print(np.array(poses_valid).shape)
    # print(np.array(poses_valid_2d).shape)
    poses_valid_2d=data_process(poses_valid_2d)
    poses_valid = data_process1(poses_valid)
    poses_valid=tf.convert_to_tensor(poses_valid)
    poses_valid_2d=tf.convert_to_tensor(poses_valid_2d)
    dataest = tf.data.Dataset.from_tensor_slices((poses_valid_2d, poses_valid))
    dataest = dataest.shuffle(buffer_size=10000).prefetch(tf.data.experimental.AUTOTUNE).repeat(
        1).batch(batch_size)
    # for epochs in range(epoch):
    #     # 使用tqdm提示训练进度
    #     with tqdm(total=2345/batch_size,desc='Epoch {}/{}'.format(epochs, epoch)) as pbar:
    #         # 每个epoch训练settings.STEPS_PER_EPOCH次
    #         for x, y in  dataest:
    #             print(x,y)
    #             loss2 = train_one_step(x, y)
    #             pbar.set_postfix(loss='%.4f' % float(loss2), acc=float(acc_meter.result()))
    #             pbar.update(1)
    model.fit(poses_valid_2d,
              poses_valid,
              batch_size=batch_size,
              epochs=epoch)
    #模型保存
    if output:
        if (os.path.exists(output) == False):
            os.makedirs(output)
        model.save(output+str(datetime.datetime.now())+".h5")
        model.save(output+str(datetime.datetime.now()), save_format='tf')
