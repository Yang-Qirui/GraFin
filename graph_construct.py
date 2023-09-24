import numpy as np
from collections import defaultdict

def LDPL(rssi, r0, n):
    return 1 / (1 + np.power(10, np.abs(rssi - r0) / (10 * n)))

def gen_rp2rp_adjacent(heat_positions, thres_k):
    np_positions = np.array(heat_positions)
    diff = np_positions[:, np.newaxis, :] - np_positions[np.newaxis, :, :]
    euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    N = euclidean_dist.shape[0]
    
    rp2rp_adjacent_matrix = np.zeros((N, N))
    
    for i in range(N):
        euclidean_dist[i, i] = np.inf # mask diagonal elements to avoid self-neighbor
        row = euclidean_dist[i, :]
        sorted_indices = np.argsort(row)
        nearest_k_rps = sorted_indices[:thres_k]
        
        # Update the adjacency matrix for the k nearest neighbors
        rp2rp_adjacent_matrix[i, nearest_k_rps] = 1 / (1 + row[nearest_k_rps])
        
    return rp2rp_adjacent_matrix

def gen_rp2ap_adjacent(pos_rssi_dict, all_aps, thres_q, r0, n):
    # new_pos_rssi_dict = {}
    # all_aps = set()
    # for rp, ap_list in pos_rssi_dict.items():
    #     sum_dict = defaultdict(float)
    #     count_dict = defaultdict(int)
    #     for ap_mac, rssi in ap_list:
    #         all_aps.add(ap_mac)
    #         sum_dict[ap_mac] += float(rssi)
    #         count_dict[ap_mac] += 1
    #     new_ap_list = [(a, sum_dict[a] / count_dict[a]) for a in sum_dict] # for a RP, it's possible for it to scan the same AP several times
    #     new_pos_rssi_dict[rp] = new_ap_list
    # all_aps = list(all_aps)
    rp_neighbor_aps_matrix = []
    for rp, ap_list in pos_rssi_dict.items():
        rp_neighbor_aps_tensor = np.full((len(all_aps)), np.NINF)
        for ap_mac, rssi in ap_list:
            ap_id = all_aps.index(ap_mac)
            rp_neighbor_aps_tensor[ap_id] = rssi
        rp_neighbor_aps_matrix.append(rp_neighbor_aps_tensor)
    rp_neighbor_aps_matrix = np.array(rp_neighbor_aps_matrix)
    for j in range(rp_neighbor_aps_matrix.shape[1]):
        column = rp_neighbor_aps_matrix[:, j]
        valid_values = column[column != np.NINF] # ignore -inf to get threshold
        if len(valid_values) == 0:
            continue
        threshold_value = np.percentile(valid_values, thres_q)
        rp_neighbor_aps_matrix[:, j][column < threshold_value] = np.NINF
    
    non_ninf_elem = rp_neighbor_aps_matrix[rp_neighbor_aps_matrix != np.NINF]
    non_ninf_adjacent_elem = [LDPL(rssi, r0, n) for rssi in non_ninf_elem] # LDPL model
    adjacent_matrix = np.zeros_like(rp_neighbor_aps_matrix)
    adjacent_matrix[rp_neighbor_aps_matrix != np.NINF] = non_ninf_adjacent_elem
    return adjacent_matrix


def gen_adjacent_matrix(pos_rssi_dict, all_aps, args):
    rp2rp_matrix = gen_rp2rp_adjacent(list(pos_rssi_dict.keys()), args.thres_k)
    rp2ap_matrix = gen_rp2ap_adjacent(pos_rssi_dict, all_aps, args.thres_q, args.r0, args.n)
    print("rp2rp mat shape:", rp2rp_matrix.shape)
    print("rp2ap mat shape:", rp2ap_matrix.shape)
    O = np.zeros((rp2ap_matrix.shape[1], rp2ap_matrix.shape[1]))
    upper_half = np.concatenate((rp2rp_matrix, rp2ap_matrix), axis=1)
    lower_half = np.concatenate((rp2ap_matrix.T, O), axis=1)
    A = np.vstack([upper_half, lower_half])
    print("adjacent matrix shape:", A.shape)
    return A, all_aps
    
