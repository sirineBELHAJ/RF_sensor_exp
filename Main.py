import csv
import pickle
from Training_Inference import read_configuration_return_results
import argparse
import os
import time
from sensor_control import initialize_bmi160, auto_calibrate, sensor_on, sensor_sleep




def parse_args():
    parser = argparse.ArgumentParser(description = "RF-V Inference")
    parser.add_argument("--dataset_name", type=str, default= "Epilepsy", help = "The Dataset name")
    parser.add_argument("--num_exits", type=int, help = "The number of exits")
    parser.add_argument("--proportions", type=float, help = "Data proportions", nargs="+")
    parser.add_argument("--th_combination", type=float, help = "Threshold combination", nargs="+")
    return parser.parse_args()


def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)


def write_content_to_file(file, content, header): # the content is a list of dictionaries
    writer = csv.writer(file)
    for line in content:
        row = [line[key] for key in header]
        writer.writerow(row)


if __name__ == "__main__":
    #*********************************************************************************
    t1 = time.time()
    time.sleep(5) # ************************************************************* for how long
    t2 = time.time()
    T = t2-t1 # ********************************************* save it
    print("t1: ", t1)
    print("t2: ", t2)

    args = parse_args()
    #results = read_configuration_return_results(args.dataset_name, args.num_exits, args.proportions, args.th_combination)
    # ***************************************************************************************************************

    file_path = f'Datasets/{args.dataset_name}_dataLabels.pkl'
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    data = data_dict['data']

    n_window, n_channel, n_data = data.shape

    window_len = n_data

    if args.dataset_name == 'Shoaib':
        fs_base = 50
    if args.dataset_name == 'Epilepsy':
        fs_base = 250
    if args.dataset_name == 'EMGPhysical':
        fs_base = 100
    if args.dataset_name == 'SelfRegulationSCP1':
        fs_base = 256
    if args.dataset_name == 'WESADchest':
        fs_base = 700
    if args.dataset_name == 'PAMAP2':
        fs_base = 100

    initialize_bmi160()
    print("BMI160 Initialized")
    auto_calibrate()


    results = read_configuration_return_results(window_len, fs_base, args.dataset_name, args.num_exits, args.proportions, args.th_combination, sensor_on=sensor_on,sensor_sleep=sensor_sleep)
    os.system("pkill -f 'python3 data_logger.py'")

    output_file = f'{args.dataset_name}_accuracy_results.csv'
    header = ['t_start','t1', 't2', 't3', 't4', 'total', 'true_label', 'prediction', 'correctness', 'exit_taken', 'data%', "sensor_total_on_sec", "sensor_total_off_sec"]
    with open(output_file, "w", newline="") as f1:
        add_header(f1, header)
        write_content_to_file(f1, results, header)
                