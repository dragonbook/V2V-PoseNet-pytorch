import numpy as np
import struct
import os


msra_data_config = {
    'img_width': 320,
    'img_height': 240,
    'min_depth': 100,
    'max_depth': 700,
    'fx': 241.42,
    'fy': 241.42,

    'joint_num': 21,
    'world_dim': 3,

    'db_dir': '/home/yalong/yalong/dataset/cvpr15_MSRAHandGestureDB',
    'result_dir': '',
    'center_dir': '/home/yalong/yalong/project/KeyPointsEstimation/V2V-PoseNet-pytorch/datasets/msra_center/',

    'folder_list': ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y'],
    'subject_num': 9,
    'test_model': 3,

    # Need to be initialized
    'train_size': 0,
    'test_size': 0,
}


def init_msra_data_config():
    # Compute training and testing samples size
    db_dir = msra_data_config['db_dir']
    folder_list = msra_data_config['folder_list']
    test_model = msra_data_config['test_model']
    subject_num = msra_data_config['subject_num']
    train_size = 0
    test_size = 0

    for mid in range(subject_num):
        num = 0
        for fd in folder_list:
            annot_dir = os.path.join(db_dir, 'P'+str(mid), fd, 'joint.txt')

            with open(annot_dir) as f:
                num = int(f.readline().rstrip())

            if mid == test_model: test_size += num
            else: train_size += num
            
    msra_data_config['train_size'] = train_size
    msra_data_config['test_size'] = test_size

    print('train_size: ', train_size)
    print('test_size: ', test_size)


# Initialize some fields of msra data configuration
init_msra_data_config()


def load_depthmap(filename):
    img_w = msra_data_config['img_width']
    img_h = msra_data_config['img_height']
    max_d = msra_data_config['max_depth']

    with open(filename, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I'*6, data[:6*4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f'*num_pixel, data[6*4:])

        cropped_image = np.asarray(cropped_image).reshape(bottom-top, -1)
        depth_image = np.zeros((img_h, img_w), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image
        depth_image[depth_image == 0] = max_d

        return depth_image


def pixel2world(x, y, z):
    img_w = msra_data_config['img_width']
    img_h = msra_data_config['img_height']
    fx = msra_data_config['fx']
    fy = msra_data_config['fy']

    w_x = (x - img_w / 2) * z / fx
    w_y = (img_h / 2 - y) * z / fy

    return w_x, w_y


def world2pixel(x, y, z):
    img_w = msra_data_config['img_width']
    img_h = msra_data_config['img_height']
    fx = msra_data_config['fx']
    fy = msra_data_config['fy']

    p_x = x * fx / z + img_w / 2
    p_y = img_h / 2 - y * fy / z

    return p_x, p_y


def load_data(db_type):
    db_dir = msra_data_config['db_dir']
    center_dir = msra_data_config['center_dir']
    test_model = msra_data_config['test_model']
    train_size = msra_data_config['train_size']
    test_size = msra_data_config['test_size']
    joint_num = msra_data_config['joint_num']
    world_dim = msra_data_config['world_dim']
    folder_list = msra_data_config['folder_list']
    subject_num = msra_data_config['subject_num']

    # Collect reference center points
    ref_pt_str = []

    if db_type == 'train':
        print('training data loading...')
        with open(os.path.join(center_dir, 'center_train_' + str(test_model) + '_refined.txt')) as f:
            ref_pt_str = [line.rstrip() for line in f]
    else:
        print('testing data loading...')
        with open(os.path.join(center_dir, 'center_test_' + str(test_model) + '_refined.txt')) as f:
            ref_pt_str = [line.rstrip() for line in f]

    print('#train_size: {}'.format(train_size))
    print('#test_size: {}'.format(test_size))
    print('db_type: {}'.format(db_type))
    print('#ref_pt_str: {}'.format(len(ref_pt_str)))

    sample_size = train_size if db_type == 'train' else test_size

    joint_world = np.zeros((sample_size, joint_num, world_dim))
    ref_pt = np.zeros((sample_size, world_dim))
    name = []

    file_id = 0
    frame_id = 0

    for mid in range(subject_num):
        if db_type == 'train': model_chk = (mid != test_model)
        elif db_type == 'test': model_chk = (mid == test_model)
        else: raise RuntimeError('unsupported db_type {}'.format(db_type))
        
        if model_chk:
            for fd in folder_list:
                annot_dir = os.path.join(db_dir, 'P'+str(mid), fd, 'joint.txt')

                lines = []
                with open(annot_dir) as f:
                    lines = [line.rstrip() for line in f]

                # skip first line
                for i in range(1, len(lines)):
                    # referece point
                    splitted = ref_pt_str[file_id].split()
                    if splitted[0] == 'invalid':
                        print('Warning: found invalid reference frame')
                        file_id += 1
                        continue
                    else:
                        ref_pt[frame_id, 0] = float(splitted[0])
                        ref_pt[frame_id, 1] = float(splitted[1])
                        ref_pt[frame_id, 2] = float(splitted[2])

                    # joint point
                    splitted = lines[i].split()
                    for jid in range(joint_num):
                        joint_world[frame_id, jid, 0] = float(splitted[jid * world_dim])
                        joint_world[frame_id, jid, 1] = float(splitted[jid * world_dim + 1])
                        joint_world[frame_id, jid, 2] = -float(splitted[jid * world_dim + 2])
                    
                    filename = os.path.join(db_dir, 'P'+str(mid), fd, '{:0>6d}'.format(i-1) + '_depth.bin')
                    name.append(filename)

                    frame_id += 1
                    file_id += 1
                        
    print('file_id: {}'.format(file_id))
    print('frame_id: {}'.format(frame_id))
    return joint_world, ref_pt, name


jw, rp, names = load_data('train')
print('train data: ')
print('#jw: ', jw.shape)
print('#rp: ', rp.shape)
print('#nm: ', len(names))

jw, rp, names = load_data('test')
print('test data: ')
print('#jw: ', jw.shape)
print('#rp: ', rp.shape)
print('#nm: ', len(names))



def depthmap2cloud(image):
    '''Convert depth image to depth cloud
    image: (h, w)
    output: (h, w, 3)
    '''
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    cloud = np.zeros((h, w, 3), dtype=np.float32)
    (cloud[:,:,0], cloud[:,:,1]), cloud[:,:,2] = pixel2world(x, y, image), image
    return cloud


def cloud2pixels(cloud):
    '''Convert 3d points to 2d pixels
    cloud: (n, 3)
    output: (n, 2)
    '''
    points_2d = np.zeros((cloud.shape[0], 2))
    points_2d[:, 0], points_2d[:, 1] = world2pixel(cloud[:,0], cloud[:, 1], cloud[:, 2])
    return points_2d
