import os
import os.path

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

__all__ = ['ErasorCarla']

label_name_mapping = {
    0: 'unlabeled',
    1: 'building',
    2: 'fence',
    3: 'other',
    4: 'pedstrian',
    5: 'pole',
    6: 'roadline',
    7: 'road',
    8: 'sidewalk',
    9: 'vegetation',
    10: 'vehicles',
    11: 'wall',
    12: 'trafficsign',
    13: 'sky',
    14: 'ground',
    15: 'bridge',
    16: 'railtrack',
    17: 'guardrail',
    18: 'trafficlight',
    19: 'static',
    20: 'dynamic',
    21: 'water',
    22: 'terrain',
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


def read_pc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines[11:]:
        values = [float(v) for v in line.strip().split()]
        points.append(values)
    points = np.asarray(points)

    return points

def generate_voxels(points, voxel_size=0.2):
    points = np.round(points / voxel_size) * voxel_size
    voxels = unique2D_subarray(points)
    # voxels -= voxels.min(0, keepdims=1)
    return voxels

def unique2D_subarray(a):
    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    b = np.ascontiguousarray(a.reshape(a.shape[0], -1)).view(dtype1)
    return a[np.unique(b, return_index=1)[1]]


def get_pose(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        values = [float(v) for v in line.strip().split()]
        poses.append(values)

    return poses

class ErasorCarla(dict):

    def __init__(self, root, voxel_size, num_points, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        if submit_to_server:
            super().__init__({
                'train':
                    ErasorCarlaInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          submit=True),
                'test':
                    ErasorCarlaInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='test')
            })
        else:
            super().__init__({
                'train':
                    ErasorCarlaInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=1,
                                          split='train',
                                          google_mode=google_mode),
                'test':
                    ErasorCarlaInternal(root,
                                          voxel_size,
                                          num_points,
                                          sample_stride=sample_stride,
                                          split='val')
            })


class ErasorCarlaInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True,
                 window=10):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.window = window
        self.seqs = []
        if split == 'train':
            self.seqs = [
                'scenario3', 'scenario5', 'scenario6', 'scenario8',
            ]

        elif self.split == 'val':
            self.seqs = [
                'scenario3', 'scenario5', 'scenario6', 'scenario8',
            ]
        elif self.split == 'test':
            self.seqs = [
               'scenario3', 'scenario5', 'scenario6', 'scenario8',
            ]

        self.map_files = dict()
        # self.odom_files = dict()
        self.files = []

        for seq in self.seqs:
            self.map_files[seq] = os.path.join(self.root, 'testing_map', seq, 'v0.1/map.npy')
            # self.odom_files[seq] = os.path.join(self.root, 'testing_data', seq, 'odom', 'scan', 'odometry.txt')
            seq_files = sorted(os.listdir(os.path.join(self.root, 'testing_data', seq, 'global_npz')))
            seq_files = [os.path.join(self.root, 'testing_data', seq, 'global_npz', x) for x in seq_files]
            self.files.extend(seq_files)

        # get map_data
        self.map = dict()
        for seq, map_file in self.map_files.items():
            map_ = np.load(map_file)
            # map = generate_voxels(map_, voxel_size=self.voxel_size)
            self.map[seq] = map_.astype(np.float32)
            # with open(map_file, 'rb') as b:
            #     # block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
            #     block_ = np.fromfile(b, dtype=np.float32)
            #
            # pc_ = np.round(block_[:, :3] / self.voxel_size).astype(np.int32)
            # pc_ -= pc_.min(0, keepdims=1)
            # self.map.extend(pc_)

        # # get odom files
        # self.odom = dict()
        # for seq, odom_file in self.odom_files.items():
        #     poses = get_pose(odom_file)
        #     self.odom[seq] = poses

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        # self.num_classes = list(label_name_mapping.keys())[-1] + 1
        self.num_classes = 2
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        # file_name = self.files[index]
        # scan_index = file_name.split("/")[-1]

        # get scan_data
        scan = np.load(self.files[index])
        block_ = scan['arr_0'].astype(np.float32)
        odom = scan['arr_1'].astype(np.float32)

        # get map_data
        scenario = self.files[index]
        scenario = scenario.split("/")[-3]
        map_ = self.map[scenario]
        radius = 2500
        map_ = map_[np.sum(np.square(map_[:, :3] - odom), axis=-1) < radius]

        block = np.zeros_like(block_)
        map = np.zeros_like(map_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
            map[:, :3] = np.dot(map_[:, :3], rot_mat) * scale_factor

        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)
            map[:, :3] = np.dot(map[:, :3], transform_mat)

        block[:, 3] = (block_[:, 3] == 1)
        map[:, 3] = (map_[:, 3] == 1)

        # pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        # pc_ -= pc_.min(0, keepdims=1)
        #
        # feat_ = block
        #
        # _, inds, inverse_map = sparse_quantize(pc_,
        #                                        return_index=True,
        #                                        return_inverse=True)
        #
        # if 'train' in self.split:
        #     if len(inds) > self.num_points:
        #         inds = np.random.choice(inds, self.num_points, replace=False)
        #
        # pc = pc_[inds]
        # feat = feat_[inds]
        # labels = labels_[inds]
        # lidar = SparseTensor(feat, pc)
        # labels = SparseTensor(labels, pc)
        # labels_ = SparseTensor(labels_, pc_)
        # inverse_map = SparseTensor(inverse_map, pc_)
        #
        # return {
        #     'lidar': lidar,
        #     'targets': labels,
        #     'targets_mapped': labels_,
        #     'inverse_map': inverse_map,
        #     'file_name': self.files[index]
        # }
        map_data = self.get_point_voxel(map[:, :3], map[:, 3], index)
        scan_data = self.get_point_voxel(block[:, :3], block[:, 3], index)

        return map_data
        # return map_data, scan_data

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def get_point_voxel(self, block, labels_, index):
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }


# class ErasorCarlaInternal:
#
#     def __init__(self,
#                  root,
#                  voxel_size,
#                  num_points,
#                  split,
#                  sample_stride=1,
#                  submit=False,
#                  google_mode=True,
#                  dataset='carla'):
#         if submit:
#             trainval = True
#         else:
#             trainval = False
#         self.root = root
#         self.split = split
#         self.voxel_size = voxel_size
#         self.num_points = num_points
#         self.sample_stride = sample_stride
#         self.google_mode = google_mode
#         self.dataset = dataset
#         self.seqs = []
#         if split == 'train':
#             self.seqs = [
#                 '02'
#             ]
#             if self.google_mode or trainval:
#                 # self.seqs.append('08')
#         elif self.split == 'val':
#             self.seqs = ['01']
#         elif self.split == 'test':
#             # self.seqs = [
#             #     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
#             # ]
#
#         self.map_files = dict()
#         self.files = []
#         # self.files = dict()
#
#         for seq in self.seqs:
#             self.map_files[seq] = [os.path.join(self.root, 'testing_map', seq, 'map', 'original.pcd')]
#             seq_files = sorted(os.listdir(os.path.join(self.root, 'testing_data', seq, 'global', 'scan')))
#             self.files.extend(seq_files)
#             # self.files[seq] = [os.path.join(self.root, 'testing_data', seq, 'global', 'scan', x) for x in seq_files]
#             # self.files.extend(seq_files)
#
#         if self.sample_stride > 1:
#             self.files = self.files[::self.sample_stride]
#
#         reverse_label_name_mapping = {}
#         self.label_map = np.zeros(260)
#         cnt = 0
#
#         for label_id in label_name_mapping:
#             if label_id > 250:
#                 if label_name_mapping[label_id].replace('moving-',
#                                                         '') in kept_labels:
#                     self.label_map[label_id] = reverse_label_name_mapping[
#                         label_name_mapping[label_id].replace('moving-', '')]
#                 else:
#                     self.label_map[label_id] = 255
#             elif label_id == 0:
#                 self.label_map[label_id] = 255
#             else:
#                 if label_name_mapping[label_id] in kept_labels:
#                     self.label_map[label_id] = cnt
#                     reverse_label_name_mapping[
#                         label_name_mapping[label_id]] = cnt
#                     cnt += 1
#                 else:
#                     self.label_map[label_id] = 255
#
#         self.reverse_label_name_mapping = reverse_label_name_mapping
#         self.num_classes = cnt
#         self.angle = 0.0
#
#     def set_angle(self, angle):
#         self.angle = angle
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         with open(self.files[index], 'rb') as b:
#             block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
#         block = np.zeros_like(block_)
#
#         if 'train' in self.split:
#             theta = np.random.uniform(0, 2 * np.pi)
#             scale_factor = np.random.uniform(0.95, 1.05)
#             rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
#                                 [-np.sin(theta),
#                                  np.cos(theta), 0], [0, 0, 1]])
#
#             block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
#         else:
#             theta = self.angle
#             transform_mat = np.array([[np.cos(theta),
#                                        np.sin(theta), 0],
#                                       [-np.sin(theta),
#                                        np.cos(theta), 0], [0, 0, 1]])
#             block[...] = block_[...]
#             block[:, :3] = np.dot(block[:, :3], transform_mat)
#
#         block[:, 3] = block_[:, 3]
#         pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
#         pc_ -= pc_.min(0, keepdims=1)
#
#         label_file = self.files[index].replace('velodyne', 'labels').replace(
#             '.bin', '.label')
#         if os.path.exists(label_file):
#             with open(label_file, 'rb') as a:
#                 all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
#         else:
#             all_labels = np.zeros(pc_.shape[0]).astype(np.int32)
#
#         labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)
#
#         feat_ = block
#
#         _, inds, inverse_map = sparse_quantize(pc_,
#                                                return_index=True,
#                                                return_inverse=True)
#
#         if 'train' in self.split:
#             if len(inds) > self.num_points:
#                 inds = np.random.choice(inds, self.num_points, replace=False)
#
#         pc = pc_[inds]
#         feat = feat_[inds]
#         labels = labels_[inds]
#         lidar = SparseTensor(feat, pc)
#         labels = SparseTensor(labels, pc)
#         labels_ = SparseTensor(labels_, pc_)
#         inverse_map = SparseTensor(inverse_map, pc_)
#
#         return {
#             'lidar': lidar,
#             'targets': labels,
#             'targets_mapped': labels_,
#             'inverse_map': inverse_map,
#             'file_name': self.files[index]
#         }
#
#     @staticmethod
#     def collate_fn(inputs):
#         return sparse_collate_fn(inputs)



class ErasorKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True,
                 dataset='carla'):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.dataset = dataset
        self.seqs = []
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
            if self.google_mode or trainval:
                self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

        self.files = []

        # read map_files
        for seq in self.seqs:
            map_files = [os.path.join(self.root, seq, 'velodyne', 'map.pcd')]

        # get scan_file list
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0

        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        label_file = self.files[index].replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
