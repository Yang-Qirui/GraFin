
import argparse
import random

from data_process import get_position_coordinate
from graph_construct import gen_adjacent_matrix

def main(args):
    tot_pos_rssi_dict = get_position_coordinate(args.path)
    selected_rps = random.sample(list(tot_pos_rssi_dict.keys()), args.num_rp)
    rp_pos_rssi_dict = {k: tot_pos_rssi_dict[k] for k in selected_rps}
    test_pos_rssi_dict = {k: v for k, v in tot_pos_rssi_dict.items() if k not in selected_rps}
    A = gen_adjacent_matrix(rp_pos_rssi_dict, args)
    


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", help="directory of path_data_files", default="./dataset/microsoft/F1/path_data_files", type=str)
    arg_parser.add_argument("--num_rp", help="number of RPs. Default is 400 according to the setting on Microsoft dataset", default=400, type=int) 
    arg_parser.add_argument("--thres_k", help="the threshold k to determine if 2 RPs are neighbors or not. It indicates the meters between two RPs", default=5, type=float) 
    arg_parser.add_argument("--thres_q", help="the threshold q to determine if a RP and a AP are neighbors or not. It indicates the RSSI between a RP and a AP", default=0.95, type=float) 
    arg_parser.add_argument("--r0", help="r0(dBm) in LDPL model", default=-30, type=float) 
    arg_parser.add_argument("--n", help="n in LDPL model", default=2.5, type=float) 
    args = arg_parser.parse_args()
    main(args)