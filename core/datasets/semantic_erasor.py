import os
import os.path
from tqdm import tqdm

import numpy as np
from torchsparse import SparseTensor
# from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from .collate import sparse_collate_fn

from sklearn.neighbors import KDTree

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

    def __init__(self, root, voxel_size, num_points, visualize, configs, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        if submit_to_server:
            super().__init__({
                'train':
                    ErasorCarlaInternal(root,
                                        voxel_size,
                                        num_points,
                                        visualize,
                                        sample_stride=1,
                                        split='train',
                                        submit=True,
                                        configs=configs),
                'test':
                    ErasorCarlaInternal(root,
                                        voxel_size,
                                        num_points,
                                        visualize,
                                        sample_stride=1,
                                        split='test',
                                        configs=configs)
            })
        else:
            super().__init__({
                'train':
                    ErasorCarlaInternal(root,
                                        voxel_size,
                                        num_points,
                                        visualize,
                                        sample_stride=1,
                                        split='train',
                                        google_mode=google_mode,
                                        configs=configs
                                        ),
                'test':
                    ErasorCarlaInternal(root,
                                        voxel_size,
                                        num_points,
                                        visualize,
                                        sample_stride=sample_stride,
                                        split='val',
                                        configs=configs)
            })


class ErasorCarlaInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 visualize,
                 split,
                 configs,
                 sample_stride=1,
                 submit=False,
                 google_mode=True,
                 window=1,
                 radius=50,
                 thres=0.02,):

        self.root = root
        self.split = split
        self.google_mode = google_mode
        self.visualize = visualize

        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.window = configs.erasor.window
        self.radius = configs.erasor.radius
        self.thres = configs.erasor.thres

        self.seqs = []
        self.configs = configs

        if split == 'train':
            self.seqs = [
                'scenario1', 'scenario2', 'scenario3', 'scenario4', 'scenario6', 'scenario7', 'scenario8', 'scenario9',
            ]

        elif self.split == 'val':
            self.seqs = [
                'scenario5',
            ]

        elif self.split == 'test':
            self.seqs = [
               'scenario5',
            ]

        self.map_files = dict()
        self.scan_files = []
        self.removed_scan_files = []

        # get scan and map data list
        for seq in self.seqs:
            self.map_files[seq] = os.path.join(self.root, 'map', seq, 'map.npy')
            seq_files = sorted(os.listdir(os.path.join(self.root, 'scan', seq, 'npz')))
            # filtering the seq_files if index is out of window
            # 281 -> window 10 -> window select f4, b5 (10) -> 0 ~ 3 remove, 277 ~ 281 remove -> 4 ~ 276
            # 281 -> window 11 -> window select f5, b5 (11) -> 0 ~ 4 remove, 277 ~ 281 remove -> 5 ~ 276
            seq_files = seq_files[(self.window + 1) // 2 - 1: - (1 + self.window // 2)]
            seq_files = [os.path.join(self.root, 'scan', seq, 'npz', x) for x in seq_files]
            self.scan_files.extend(seq_files)

        # get map_data
        self.map = dict()
        for seq, map_file in self.map_files.items():
            map_ = np.load(map_file)
            # map = generate_voxels(map_, voxel_size=self.voxel_size)
            # map_[:, 4] = self.revise_dynamic_points(map_[:, :3], map_[:, 4], self.thres)
            self.map[seq] = map_.astype(np.float32)

        # # exclude scan file with no overlap
        # for scan_file in sorted(self.scan_files):
        #     scan = np.load(scan_file)
        #     odom = scan['arr_1'].astype(np.float32)
        #     scenario = scan_file.split("/")[-3]
        #     map_ = self.map[scenario]
        #     map_r = map_[np.sum(np.square(map_[:, :3] - odom), axis=-1) < self.radius * self.radius]
        #     if map_r.shape[0] < self.num_points / 5:
        #     # if map_r.shape[0] == 0:
        #         self.scan_files.remove(scan_file)
        #         self.removed_scan_files.extend(scan_file)
        #         print("scan_file ", scan_file, " was removed. ")
        #         print("map_r shape: ", map_r.shape)
        #     else:
        #         print("scan_file ", scan_file, " was not removed. ")
        #         print("map_r shape: ", map_r.shape)

        if self.sample_stride > 1:
            self.scan_files = self.scan_files[::self.sample_stride]

        # self.num_classes = list(label_name_mapping.keys())[-1] + 1
        self.num_classes = 2
        self.angle = 0.0

    def revise_dynamic_points(self, points, labels, thres):
        labels[(labels != 0) & (points[:, 2] < thres)] = 0
        return labels

    def concatenate_scans(self, index):
        scan = np.load(self.scan_files[index])
        odom = scan['arr_1'].astype(np.float32)
        # 281 -> window 10 -> window select f4, b5 (10) -> 0 ~ 3 remove, 277 ~ 281 remove -> 4 ~ 276
        # 281 -> window 11 -> window select f5, b5 (11) -> 0 ~ 4 remove, 277 ~ 281 remove -> 5 ~ 276
        # index_list: index - (self.window + 1) // 2 + 1: index + self.window // 2 + 1
        file_name = os.path.basename(self.scan_files[index])
        file_dir = os.path.dirname(self.scan_files[index])
        file_name_wo_ext = os.path.splitext(file_name)[0]
        file_int = int(file_name_wo_ext)
        file_int_list = list(range(file_int - (self.window + 1) // 2 + 1, file_int + self.window // 2 + 1, 1))
        file_str_list = [file_dir + "/" + str(i).zfill(6) + ".npz" for i in file_int_list]
        block_ = []
        for file in file_str_list:
            scan_ = np.load(file)
            block_single = scan_['arr_0'].astype(np.float32)
            block_.extend(block_single)
        block_ = np.asarray(block_)
        # radius search w.r.t the odom of scan data
        block_ = block_[np.sum(np.square(block_[:, :3] - odom), axis=-1) < self.radius * self.radius]

        return block_, odom

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):

        # get scan_data
        if self.window == 1:
            scan = np.load(self.scan_files[index])
            block_ = scan['arr_0'].astype(np.float32)
            odom = scan['arr_1'].astype(np.float32)
            block_r = block_[np.sum(np.square(block_[:, :3] - odom), axis=-1) < self.radius * self.radius]

        else:
            # concatenate the scan data with window size
            block_r, odom = self.concatenate_scans(index)

        # get map_data
        scan_files = self.scan_files[index]
        scenario = scan_files.split("/")[-3]
        map_ = self.map[scenario]
        # radius search w.r.t the odom of scan data
        map_r = map_[np.sum(np.square(map_[:, :3] - odom), axis=-1) < self.radius * self.radius]
        if map_r.shape[0] < self.num_points / 10:
            print("map_r points shortage")
            print("index: ", index)
            print("scan_files: ", self.scan_files[index])

        block_T = block_r
        map_T = map_r

        if 'train' in self.split:
            # data augmentation on the train dataset
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block_T[:, :3] = np.dot(block_r[:, :3], rot_mat) * scale_factor
            map_T[:, :3] = np.dot(map_r[:, :3], rot_mat) * scale_factor

        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block_T[:, :3] = np.dot(block_r[:, :3], transform_mat)
            map_T[:, :3] = np.dot(map_r[:, :3], transform_mat)

        # parsing the original label to the dynamic label
        block_T[:, 3:] = (block_r[:, 3:] != 0)
        map_T[:, 3:] = (map_r[:, 3:] != 0)

        # get point and voxel in the format of sparse torch tensor
        map_data = self.get_point_voxel(map_T, index)
        # scan_data = self.get_point_voxel(block_T, index)

        return map_data
        # return map_data, scan_data

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def get_point_voxel(self, points, index):
        # points_ -> pc_ -> pc, labels_ -> labels, proposals_ -> proposals
        if np.shape(points)[-1] == 6:
            points_, labels_, proposals_ = points[:, :3], points[:, 3], points[:, 4]
        elif np.shape(points)[-1] == 4:
            points_, labels_, proposals_ = points[:, :3], points[:, 3], None

        # voxelization & get inds
        try:
            pc_ = np.round(points_ / self.voxel_size).astype(np.int32)
            pc_ -= pc_.min(0, keepdims=1)
        except:
            print("pc_.shape: ", pc_.shape)
            print("scan_file: ", self.scan_files[index])

        feat_ = points_

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
        targets = SparseTensor(labels, pc)
        targets_mapped = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        if proposals_ is not None:
            proposals = proposals_[inds]
            proposals = SparseTensor(proposals, pc)
            proposals_ = SparseTensor(proposals_, pc_)
            """
            if int(self.configs.erasor.knn) > 0:
                proposals = self.propose_near_dynamic_points(pc,
                                                             proposals,
                                                             self.configs.erasor.leaf_size,
                                                             self.configs.erasor.k)
            """
        else:
            proposals = proposals_

        if ('train' in self.split) and (self.configs.erasor.erasor_proposal) and (proposals_ != None):
            targets = proposals
            targets_mapped = proposals_

        feed_dict = {
            'points_': points_,
            'pc_': pc_,
            'inds': inds,
            'pc': pc,
            'feat': feat,
            'labels': labels,
            'lidar': lidar,
            'targets': targets,
            'targets_mapped': targets_mapped,
            'proposals': proposals,
            'inverse_map': inverse_map,
            'file_name': self.scan_files[index]
        }

        return feed_dict

    def save_points(self, points, labels, label2color, path):
        #     points = (points - points.mean()) / points.std()
        import open3d as o3d
        colors = np.array([label2color[x] for x in labels])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(path, pcd)

    def propose_near_dynamic_points(self, points, labels, leaf_size=40, k=10):
        """propose top k near points from dynamic points"""
        # nearest point search intialization
        tree = KDTree(points, leaf_size=leaf_size)
        dist, ind = tree.query(points, k=k)
        # index of the top 5 near points from dynamic points -> set -> label the point as dynamic point
        labels[np.unique(ind[labels == 1])] = 1
        return labels


class ErasorKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 visualize,
                 split,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
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
        self.visualize = visualize
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
