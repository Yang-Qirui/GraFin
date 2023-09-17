from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.append("./dataset/microsoft")
from dataset.microsoft.main import calibrate_magnetic_wifi_ibeacon_to_position, extract_wifi_count, extract_wifi_rssi

def get_position_coordinate(path_data_dir):
    '''
        this function returns: 
        {RP coordinate: [[AP1 MAC, RSSI1], [AP2 MAC, RSSI2]...]}
    '''
    path_filenames = list(Path(path_data_dir).resolve().glob("*.txt"))
    mwi_datas = calibrate_magnetic_wifi_ibeacon_to_position(path_filenames)
    wifi_counts = extract_wifi_count(mwi_datas)
    heat_positions = np.array(list(wifi_counts.keys()))
    heat_values = np.array(list(wifi_counts.values()))
    # filter out positions that no wifi detected
    mask = heat_values != 0
    heat_positions = heat_positions[mask]
    heat_values = heat_values[mask]
    print("heat_positionscount count:", len(heat_positions))
    pos_rssi_dict = {}
    for heat_pos in heat_positions.tolist():
        pos_rssi_dict[tuple(heat_pos)] = tuple(mwi_datas[tuple(heat_pos)]['wifi'][:, 2:4])
    return pos_rssi_dict 

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", help="directory of path_data_files", default="./dataset/microsoft/F1/path_data_files", type=str)
    args = arg_parser.parse_args()
    get_position_coordinate(args.path)