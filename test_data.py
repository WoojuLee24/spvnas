import argparse
import sys
import open3d as o3d
import numpy as np
import os
import os.path
import shutil
import numpy as np
import random

from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder

from sklearn.neighbors import KDTree
# from core.datasets.semantic_erasor import ErasorCarlaInternal


def save_points(points, labels, label2color, path):
    # points = (points - points.mean()) / points.std()
    colors = np.array([label2color[x] for x in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def pass_points(points):
    return points

def translate(points):
    points_T = points.copy()
    points_T[:, :3] = points_T[:, :3] + 0.01
    return points_T

def rotate(points):
    random.seed()
    theta = np.pi / 5 * random.random()
    points_T = points.copy()
    points_mean = points.mean(axis=0, keepdims=True)[:, :3]
    points_std = points.std(axis=0, keepdims=True)[:, :3]
    points_T[:, :3] = (points[:, :3] - points_mean) / points_std
    transform_mat = np.array([[np.cos(theta),
                               np.sin(theta), 0],
                              [-np.sin(theta),
                               np.cos(theta), 0], [0, 0, 1]])
    points_T[:, :3] = np.dot(points_T[:, :3], transform_mat)
    points_T[:, :3] = points_T[:, :3] * points_std + points_mean
    return points_T


def shear_z(points, sx=0.01, sy=0.01):
    random.seed()
    points_T = points.copy()
    arr = [-1, 0, 1]
    sx *= random.choice(arr)
    sy *= random.choice(arr)
    points_mean = points.mean(axis=0, keepdims=True)[:, :3]
    points_std = points.std(axis=0, keepdims=True)[:, :3]
    points_T[:, :3] = (points[:, :3] - points_mean) / points_std
    transform_mat = np.array([[1, 0, sx],
                              [0, 1, sy],
                              [0, 0, 1]])
    points_T[:, :3] = np.dot(points_T[:, :3], transform_mat)
    points_T[:, :3] = points_T[:, :3] * points_std + points_mean

    return points_T


def shear_xy(points, s=0.1):
    random.seed()
    points_T = points.copy()
    arr = [-1, 0, 1]
    sxy = s * random.choice(arr)
    syx = s * random.choice(arr)
    szx = s * random.choice(arr)
    szy = s * random.choice(arr)
    points_mean = points.mean(axis=0, keepdims=True)[:, :3]
    points_std = points.std(axis=0, keepdims=True)[:, :3]
    points_T[:, :3] = (points[:, :3] - points_mean) / points_std
    transform_mat = np.array([[1, sxy, 0],
                              [syx, 1, 0],
                              [szx, szy, 1]])
    points_T[:, :3] = np.dot(points_T[:, :3], transform_mat)
    points_T[:, :3] = points_T[:, :3] * points_std + points_mean

    return points_T

def get_partition(points, partitions=4):
    x_max, x_min, y_max, y_min, z_max, z_min = \
        points[:, 0].max(), points[:, 0].min(), \
        points[:, 1].max(), points[:, 1].min(), \
        points[:, 2].max(), points[:, 2].min()


def make_points_sparse(points, ratio=0.7, num_points_limit=1000, distance_limit=100, p=1.0, FPS=False, gt_box_idx=None):
    # static to static for robustness
    num_points = points.shape[0]
    sparse_point_idx = np.random.choice(range(points.shape[0]), int(num_points*ratio), replace=False)
    sparse_points = points[sparse_point_idx]
    return sparse_points

def jitter(points, sigma=0.01, p=0.5, distance_limit=100, gt_box_idx=None):
    # translate poitns with gaussian noise
    # static to static for robustness
    translation_noise = np.random.normal(0, sigma, size=points.shape)
    if np.random.rand(1) > p:
        pass
    else:
        points += translation_noise

    return points


class ErasorCarlaInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 #visualize,
                 split,
                 configs,
                 sample_stride=1,
                 #submit=False,
                 #google_mode=True,
                 #window=1,
                 #radius=50,
                 #thres=0.02,
                 ):

        self.root = root
        self.split = split
        #self.google_mode = google_mode
        #self.visualize = visualize

        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.window = configs.erasor.window
        self.radius = configs.erasor.radius
        self.thres = configs.erasor.thres

        # debug
        if configs.dataset.debug == True:
            self.save_path = configs.dataset.save_path
            self.index = configs.dataset.index
        else:
            self.save_path = None
            self.index = -1

        self.seqs = []
        self.configs = configs
        if configs.erasor.aug == 'all':
            self.aug_list = [rotate, translate, make_points_sparse, jitter, shear_z, shear_xy]
        elif configs.erasor.aug == 'rotate':
            self.aug_list = [rotate]
        elif configs.erasor.aug == 'none':
            self.aug_list = [pass_points]



        if split == 'train':
            self.seqs = [
                'scenario1', 'scenario2', 'scenario4', 'scenario6', 'scenario7', 'scenario8', 'scenario9',
                # 'scenario3'
            ]

        elif self.split == 'val':

            self.seqs = [
                'scenario3',
                # 'scenario2',
            ]


        elif self.split == 'test':
            self.seqs = [
                'scenario3',
                # 'scenario2',
            ]

        self.map_files = dict()
        self.scan_files = []
        self.removed_scan_files = []

        # get scan and map data list
        for seq in self.seqs:
            self.map_files[seq] = os.path.join(self.root, 'map', seq, 'map_cluster.npy')
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

        # # check scan file with no overlap
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
        # debug block_r points
        block = scan['arr_0'].astype(np.float32)
        block_r = block[np.sum(np.square(block[:, :3] - odom), axis=-1) < self.radius * self.radius]

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

        if self.index != -1:
            index = self.index

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
        # debug
        a = np.sum(map_[:, 4])
        b = np.shape(map_[:, 4])[0]
        # radius search w.r.t the odom of scan data
        map_r = map_[np.sum(np.square(map_[:, :3] - odom), axis=-1) < self.radius * self.radius]
        # debug
        a = np.sum(map_r[:, 4])
        b = np.shape(map_r[:, 4])[0]
        # debug
        if map_r.shape[0] < self.num_points / 5:
            print("map_r points shortage")
            print("index: ", index)
            print("scan_files: ", self.scan_files[index])

        block_T = block_r
        map_T = map_r

        if 'train' in self.split:
            # # data augmentation on the train dataset
            # theta = np.random.uniform(0, 2 * np.pi)
            # scale_factor = np.random.uniform(0.95, 1.05)
            # rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
            #                     [-np.sin(theta),
            #                      np.cos(theta), 0], [0, 0, 1]])
            #
            # block_T[:, :3] = np.dot(block_r[:, :3], rot_mat) * scale_factor
            # map_T[:, :3] = np.dot(map_r[:, :3], rot_mat) * scale_factor
            # debug
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block_T[:, :3] = np.dot(block_r[:, :3], transform_mat)
            map_T[:, :3] = np.dot(map_r[:, :3], transform_mat)

        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block_T[:, :3] = np.dot(block_r[:, :3], transform_mat)
            map_T[:, :3] = np.dot(map_r[:, :3], transform_mat)

        # parsing the original label to the dynamic label for the gt and proposals
        block_T[:, 3:] = (block_r[:, 3:] != 0)
        # map_T[:, 3] = (map_r[:, 3:] != 0)
        map_T[:, 3] = (map_r[:, 3] != 0)
        map_T[:, 4] = (map_r[:, 4] != 0)
        if ('train' in self.split) and (configs.erasor.cluster > 0) and (configs.erasor.map_only == False):
            map_clusters, map_cluster_inds = self.choose_cluster(map_T, n=5)
            block_clusters, block_cluster_inds, map_dynamic_cluster_ind = self.get_cluster_inds(map_clusters, map_cluster_inds, block_T)
            # self-labeling dynamic clusters in the map.
            a = np.sum(map_T[:, 4])
            map_T[:, 4] += map_dynamic_cluster_ind.astype(np.float32) * 2
            # map_T[:, 4] = (map_T[:, 4] != 0)
            b = np.sum(map_T[:, 4])
            for ind in block_cluster_inds:
                # b = self.augment_points(block_T[ind])
                block_T[ind] = self.augment_points(block_T[ind])
                # block_T[ind] = rotate(block_T[ind])
                # block_T[ind] = shear_z(block_T[ind])
                # block_T[ind] = shear_xy(block_T[ind])

        # get point and voxel in the format of sparse torch tensor
        if configs.erasor.map_only == False:
            map_data = self.get_point_voxel(map_T, index)
            scan_data = self.get_point_voxel(block_T, index)
            return map_data, scan_data

        else:
            map_data = self.get_point_voxel(map_T, index)
            return map_data


    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)

    def choose_cluster(self, points, n):
        # choose map clusters for proposed static points
        static_points = points[points[:, 4] == 0]
        # get cluster indexes
        a = np.unique(points[:, 6])
        cluster_class = np.unique(static_points[:, 6])
        cluster_class = cluster_class[cluster_class != 0]    # 0 is not cluster.
        if n > len(cluster_class):
            n = len(cluster_class)
        selected_cluster_class = np.random.choice(cluster_class, size=n, replace=False)
        # get clusters for chosen cluster.
        # Chosen clusters can contain dyanmic points connected to the static poitns
        clusters = []
        cluster_inds = []
        for i in selected_cluster_class:
            cluster_ind = points[:, 6] == i
            cluster = points[cluster_ind]
            # prevent small clusters: scan often does not contain small objects in the map.
            if np.shape(cluster)[0] > 100:
                cluster_inds.append(cluster_ind)
                clusters.append(cluster)
        return clusters, cluster_inds

    def get_cluster_inds(self, clusters, cluster_inds, block):
        # map cluster -> range -> scan cluster index
        block_clusters = []
        block_cluster_inds = []
        map_cluster_ind = np.full(np.shape(cluster_inds[0]), False)
        for i, cluster in enumerate(clusters):
            # get cluster range of map data
            x_max, x_min, y_max, y_min, z_max, z_min = \
                cluster[:, 0].max(), cluster[:, 0].min(), \
                cluster[:, 1].max(), cluster[:, 1].min(), \
                cluster[:, 2].max(), cluster[:, 2].min()
            # index cluster range with x and y coordinate,
            # because ground segmentation is not very accurate
            ind = (x_min <= block[:, 0]) & (block[:, 0] <= x_max) & \
                  (y_min <= block[:, 1]) & (block[:, 1] <= y_max)
            block_cluster = block[ind]
            if np.shape(block_cluster)[0] > 20:
                block_clusters.append(block_cluster)
                block_cluster_inds.append(ind)
                map_cluster_ind = map_cluster_ind | cluster_inds[i]
                k = np.sum(map_cluster_ind.astype(np.float32))
                a = k
        # k = np.sum(map_cluster_ind.astype(int))
        return block_clusters, block_cluster_inds, map_cluster_ind

    def augment_points(self, block):
        # augment cluster of block (scan)
        op = np.random.choice(self.aug_list)
        augmented_block = op(block)

        return augmented_block


    def get_point_voxel(self, points, index):
        # points_ -> pc_ -> pc, labels_ -> labels, proposals_ -> proposals
        if np.shape(points)[-1] == 8:
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

        # if 'train' in self.split:
        #     if len(inds) > self.num_points:
        #         inds = np.random.choice(inds, self.num_points, replace=False)

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

            target_proposals = SparseTensor(proposals, pc)
            target_proposals_ = SparseTensor(proposals_, pc_)

        else:
            proposals = proposals_
            target_proposals = proposals
            target_proposals_ = proposals_

        # if ('train' in self.split) and (self.configs.erasor.erasor_proposal) and (proposals_ != None):
        #     targets = proposals
        #     targets_mapped = proposals_

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
            'target_proposals': target_proposals,
            'target_proposals_': target_proposals_,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    split = 'train'
    # dataset = builder.make_dataset()
    dataset = ErasorCarlaInternal(root=configs.dataset.root,
                                  voxel_size=configs.dataset.voxel_size,
                                  num_points=configs.dataset.num_points,
                                  sample_stride=1,
                                  split=split,
                                  configs=configs)

    if configs.erasor.map_only:
        map_data = next(iter(dataset))
        label2color = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 1.0, 1.0)}
        mpoints = map_data['feat']
        mlabels = map_data['labels']
        mproposals = map_data['proposals']

        aug_case = configs.erasor.aug

        path = "/ws/data/erasor_carla/debug/map3_index10_12k_cluster_" + aug_case + "_" + split + ".pcd"
        save_points(mpoints, mlabels, label2color, path)

        path = "/ws/data/erasor_carla/debug/map3_index10_12k_cluster_proposal_" + aug_case + "_" + split + ".pcd"
        save_points(mpoints, mproposals, label2color, path)

    else:

        map_data, scan_data = next(iter(dataset))

        label2color = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 1.0, 1.0)}
        mpoints = map_data['feat']
        mlabels = map_data['labels']
        mproposals = map_data['proposals']


        a = np.sum(mproposals==0.0)
        b = np.sum(mproposals==1.0)
        c = np.sum(mproposals==2.0)
        d = np.sum(mproposals==3.0)

        points = scan_data['feat']
        labels = scan_data['labels']

        aug_case = configs.erasor.aug

        path = "/ws/data/erasor_carla/debug/map3_index10_12k_cluster_" + aug_case + "_" + split + ".pcd"
        save_points(mpoints, mlabels, label2color, path)

        path = "/ws/data/erasor_carla/debug/map3_index10_12k_cluster_proposal_" + aug_case + "_" + split + ".pcd"
        save_points(mpoints, mproposals, label2color, path)

        path = "/ws/data/erasor_carla/debug/scan3_index10_12k_w10_" + aug_case + "_" + split + ".pcd"
        save_points(points, labels, label2color, path)


if __name__ == '__main__':
    main()