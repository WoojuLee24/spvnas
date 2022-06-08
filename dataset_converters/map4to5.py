import os
import numpy as np

root_path = "/ws/data/erasor_carla/carla_v0.2/map"

scenarioes = os.listdir(root_path)
for scenario in scenarioes:
    scenario_path = os.path.join(root_path, scenario)
    gt_path = os.path.join(scenario_path, "gt.npy")
    erasor_path = os.path.join(scenario_path, "erasor.npy")
    new_map_path = os.path.join(scenario_path, "map.npy")
    gt = np.load(gt_path)
    erasor = np.load(erasor_path)
    # gt_sort = np.sort(gt, axis=0)
    # erasor_sort = np.sort(erasor, axis=0)
    gt_ind = np.lexsort((gt[:, 2], gt[:, 1], gt[:, 0]))
    gt_sort = gt[gt_ind]

    erasor_ind = np.lexsort((erasor[:, 2], erasor[:, 1], erasor[:, 0]))
    erasor_sort = erasor[erasor_ind]

    # k = np.array_equal(erasor_sort//0.1*0.1, gt_sort//0.1*0.1)
    # t = erasor_sort[:, -1:]
    new_map = np.concatenate((gt_sort, erasor_sort[:, -1:]), axis=1)
    np.save(new_map_path, new_map)
