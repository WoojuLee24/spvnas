import os
import numpy as np
import open3d as o3d

def read_pc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines[11:]:
        values = [float(v) for v in line.strip().split()]
        points.append(values)
    points = np.asarray(points)

    return points


def read_pc_with_o3d(path):
    pcd = o3d.io.read_point_cloud(path)
    print(np.asarray(pcd.points))
    return np.asarray(pcd.points)


def get_pose(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        values = [float(v) for v in line.strip().split()]
        poses.append(values)

    return poses

# seqs = ['scenario1', 'scenario2', 'scenario3', 'scenario4', 'scenario5', 'scenario6', 'scenario7', 'scenario8']
seqs = ['scenario8', 'scenario9']
root_path = "/ws/data/erasor_carla/carla_v0.2/"

# save map npy data
for seq in seqs:
    print(seq)
    map_path = os.path.join(root_path, "map", seq, "map.pcd")
    new_map_path = os.path.splitext(map_path)[0] + ".npy"
    points = read_pc(map_path)
    np.save(new_map_path, points)


# # save scan npy data
# for seq in seqs:
#     scan_dir = os.path.join(root_path, "scan", seq, "pcd")
#     new_scan_dir = os.path.join(root_path, "scan", seq, "npz")
#     odom_path = os.path.join(root_path, "scan", seq, "odom.txt")
#     odoms = get_pose(odom_path)
#     if not os.path.exists(new_scan_dir):
#         os.makedirs(new_scan_dir)
#         # os.makedirs(os.path.join(new_scan_dir, "scan"))
#         # new_scan_dir = os.path.join(new_scan_dir, "scan")
#     scan_filenames = sorted(os.listdir(scan_dir))
#     for idx, scan_filename in enumerate(scan_filenames):
#         scan_file = os.path.join(scan_dir, scan_filename)
#         points = read_pc(scan_file)
#         odom = odoms[idx]
#         new_scan_file = os.path.join(new_scan_dir, scan_filename[:-4])
#         np.savez(new_scan_file, points, odom)
#         # np.save(new_scan_file, points)

