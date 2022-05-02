import os
import numpy as np

def read_pc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines[11:]:
        values = [float(v) for v in line.strip().split()]
        points.append(values)
    points = np.asarray(points)

    return points

def get_pose(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        values = [float(v) for v in line.strip().split()]
        poses.append(values)

    return poses

# seqs = ['scenario1', 'scenario2', 'scenario3', 'scenario5', 'scenario6', 'scenario8']

seqs = ['scenario2', 'scenario3', 'scenario5', 'scenario6', 'scenario8']
# seqs = ['scenario3', 'scenario5', 'scenario6', 'scenario8']
root_path = "/ws/data/erasor_carla/carla_dataset/"

# # save map npy data
# for seq in seqs:
#     print(seq)
#     map_path = os.path.join(root_path, "testing_map/v0.1", seq, "map.pcd")
#     new_map_path = os.path.splitext(map_path)[0] + ".npy"
#     points = read_pc(map_path)
#     np.save(new_map_path, points)

# save scan npy data
for seq in seqs:
    scan_dir = os.path.join(root_path, "testing_data", seq, "global/scan/")
    new_scan_dir = os.path.join(root_path, "testing_data", seq, "global_npz")
    odom_path = os.path.join(root_path, "testing_data", seq, "odom/scan/odometry.txt")
    odoms = get_pose(odom_path)
    if not os.path.exists(new_scan_dir):
        os.makedirs(new_scan_dir)
        # os.makedirs(os.path.join(new_scan_dir, "scan"))
        # new_scan_dir = os.path.join(new_scan_dir, "scan")
    scan_filenames = sorted(os.listdir(scan_dir))
    scan_files = [os.path.join(scan_dir, x) for x in scan_filenames]
    dummy = scan_files
    for idx, scan_filename in enumerate(scan_filenames):
        scan_file = os.path.join(scan_dir, scan_filename)
        points = read_pc(scan_file)
        odom = odoms[idx]
        new_scan_file = os.path.join(new_scan_dir, scan_filename[:-4])
        np.savez(new_scan_file, points, odom)
        # np.save(new_scan_file, points)


### deprecated
# # save map pose npy
# for seq in seqs:
#     odom_path = os.path.join("/ws/data/erasor_carla/carla_dataset/testing_data", seq, "odom/map/odometry.txt")
#     new_odom_path = os.path.splitext(odom_path)[0] + ".npy"
#     points = read_pc(odom_path)
#     np.save(new_odom_path, points)

# # save scan pose npy
# for seq in seqs:
#     odom_path = os.path.join("/ws/data/erasor_carla/carla_dataset/testing_data", seq, "odom/scan/odometry.txt")
#     new_odom_path = os.path.splitext(odom_path)[0] + ".npy"
#     points = read_pc(odom_path)
#     np.save(new_odom_path, points)