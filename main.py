
import argparse
import random
import time
import torch
from tqdm import tqdm
import numpy as np

from data_process import get_position_coordinate, aggregate_duplicate_AP_scan
from graph_construct import gen_adjacent_matrix, LDPL
from feature_extract import gen_landmarks_features
from model import GraFinModel, GraFinLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_k_closest_rows(query_vector, matrix, k=1):
    distances = torch.norm(matrix - query_vector, dim=1)
    _, indices = torch.topk(distances, k, largest=False)
    weights = 1 / (1 + distances[indices])
    norm_weights = weights / torch.sum(weights)
    return norm_weights, indices.tolist()

def run(args, adjacent_matrix, features, test_set, all_aps, rp_pos):
    rp2rp_adjacent = adjacent_matrix[:args.num_rp, :args.num_rp] # rp x rp
    D_rp2rp = torch.diag(torch.sum(rp2rp_adjacent, dim=1)).to(device) # degree matrix
    pad_matrix = torch.zeros_like(adjacent_matrix)
    pad_matrix[:args.num_rp, :args.num_rp] = rp2rp_adjacent
    rp2ap_adjacent = adjacent_matrix - pad_matrix # (rp + ap) x (rp + ap)
    D_rp2ap = torch.diag(torch.sum(rp2ap_adjacent, dim=1)).to(device) # degree matrix

    L_rp2ap = D_rp2ap - rp2ap_adjacent
    L_rp2rp = D_rp2rp - rp2rp_adjacent

    model = GraFinModel(2, args.num_rp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = GraFinLoss(args.alpha)

    min_loss, min_err = float('inf'), float('inf')
    with tqdm(total=args.epoch) as t:
        for e in range(args.epoch):
            # Forward phase
            Y = model(adjacent_matrix, features)
            loss = loss_fn(Y, D_rp2rp, L_rp2rp, D_rp2ap, L_rp2ap)

            # Back propagate phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            ap_fps = Y[args.num_rp:]
            rp_fps = torch.mm(adjacent_matrix[:args.num_rp, args.num_rp:], ap_fps)
            test_pbar = tqdm(test_set.items(), total=len(test_set), desc="Testing", leave=False)
            cnt, err = 0, 0
            for k, v in test_pbar:
                ground_truth_coordinate = torch.tensor(k).unsqueeze(0)
                ap_list = list(v)
                rp_fp = torch.zeros((1, rp_fps.shape[1]))
                for ap_mac, ap_rssi in ap_list:
                    id = all_aps.index(ap_mac)
                    rp_fp += 1 / (1 + LDPL(ap_rssi, args.r0, args.n)) * ap_fps[id]
                norm_weights, indices = find_k_closest_rows(rp_fp, rp_fps, args.top_k)
                predict_pos = torch.mm(norm_weights.unsqueeze(0), rp_pos[indices])
                err += torch.nn.functional.l1_loss(predict_pos, ground_truth_coordinate)
                cnt += 1

                test_pbar.set_postfix(test_error=err.item() / cnt)
                test_pbar.update(1)

            t.set_description(f"Epoch {e}")            
            t.set_postfix(loss=loss.item(), error=err.item() / cnt)
            if loss.item() < min_loss:
                min_loss = loss.item()
            if err.item() < min_err:
                min_err = err.item()
            t.set_postfix(min_loss=min_loss, min_error=min_err)
            t.update(1)

def main(args):
    torch.autograd.set_detect_anomaly(True)
    tot_pos_rssi_dict = get_position_coordinate(args.path)
    tot_pos_rssi_dict, all_aps = aggregate_duplicate_AP_scan(tot_pos_rssi_dict)
    selected_rps = random.sample(list(tot_pos_rssi_dict.keys()), args.num_rp)
    rp_pos_rssi_dict = {k: tot_pos_rssi_dict[k] for k in selected_rps}
    test_pos_rssi_dict = {k: v for k, v in tot_pos_rssi_dict.items() if k not in selected_rps}
    A = gen_adjacent_matrix(rp_pos_rssi_dict, all_aps, args)
    X = gen_landmarks_features(A, args.thres_hop, args.num_rp)
    start = time.time()
    run(args, torch.from_numpy(A).to(device), torch.from_numpy(X).to(device), test_set=test_pos_rssi_dict, all_aps=all_aps, rp_pos=torch.tensor(list(rp_pos_rssi_dict.keys()), dtype=torch.double).to(device))
    end = time.time()
    print(end - start)



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", help="directory of path_data_files", default="./dataset/microsoft/F1/path_data_files", type=str)
    arg_parser.add_argument("--num_rp", help="number of RPs. Default is 400 according to the setting on Microsoft dataset", default=400, type=int) 
    arg_parser.add_argument("--thres_k", help="the threshold k to determine if 2 RPs are neighbors or not. It indicates the meters between two RPs", default=5, type=float) 
    arg_parser.add_argument("--thres_q", help="the threshold q to determine if a RP and a AP are neighbors or not. It indicates the RSSI between a RP and a AP", default=0.95, type=float) 
    arg_parser.add_argument("--r0", help="r0(dBm) in LDPL model", default=-30, type=float) 
    arg_parser.add_argument("--n", help="n in LDPL model", default=2.5, type=float)
    arg_parser.add_argument("--thres_hop", help="the threshold hop to control the maximum hop number", default=1, type=int) 
    arg_parser.add_argument("--epoch", help="training epoch number", default=2500, type=int)
    arg_parser.add_argument("--lr", help="training learning rate", default=0.1, type=float)
    arg_parser.add_argument("--top_k", help="top k nearest rp", default=3, type=int)
    arg_parser.add_argument("--alpha", help="alpha to adjust loss", default=0.5, type=float)


    args = arg_parser.parse_args()
    main(args)