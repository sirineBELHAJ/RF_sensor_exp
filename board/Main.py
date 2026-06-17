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
    args = parse_args()
    # t1 = time.time()
    # time.sleep(5) # ************************************************************* for how long
    # t2 = time.time()
    # T = t2-t1 # ********************************************* save it
    # p = "start_end_times.txt"
    # with open(p, 'a') as f:
    #     f.write(f"{args.dataset_name}_start: {t1}\n")
    #     f.write(f"{args.dataset_name}_end: {t2}\n")


    # print("t1: ", t1)
    # print("t2: ", t2)

    
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
        window_T = 4
        l = [
                2, 10, 24, 29, 30, 31, 39, 41, 49, 55, 56, 60, 70, 76, 77, 78, 82, 86, 90,
                101, 110, 131, 132, 135, 145, 148, 155, 158, 163, 165, 174, 176, 188, 192,
                209, 210, 211, 213, 215, 228, 244, 247, 256, 257, 261, 264, 265, 268, 274,
                275, 278, 281, 286, 289, 290, 298, 304, 316, 318, 331, 338, 341, 350, 357,
                363, 364, 396, 403, 404, 405, 412, 414, 417, 453, 456, 457, 470, 473, 477,
                479, 487, 494, 497, 506, 531, 539, 549, 552, 571, 584, 587, 593, 598, 599,
                605, 610, 613, 617, 625, 626
                ]
    if args.dataset_name == 'Epilepsy':
        fs_base = 250
        window_T = 0.82
        l = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
            ]
    if args.dataset_name == 'EMGPhysical':
        fs_base = 100
        window_T = 2
        l = [
            0, 4, 5, 9, 10, 11, 12, 15, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 35, 36, 39, 40, 41, 42, 44, 45, 47, 49, 51, 53, 55, 56, 60, 61,
            62, 64, 65, 66, 67, 68, 69, 73, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 90,
            91, 94, 95, 96, 97, 98, 99, 102, 105, 106, 109, 110, 111, 112, 113, 115, 118,
            119, 120, 123, 124, 125, 126, 127, 128, 129, 132, 133, 135, 136, 139, 140,
            143, 144, 145, 146, 148, 149, 150, 152, 153, 155
            ]
    if args.dataset_name == 'SelfRegulationSCP1':
        fs_base = 256
        window_T = 3.5
        l = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62,
            63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 75, 76, 77, 78, 79, 80, 81, 83, 84,
            85, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 101, 102, 104, 105, 106,
            107, 109, 110, 111
            ]
    if args.dataset_name == 'WESADchest':
        fs_base = 700
        window_T = 0.28
        l = [
            23, 29, 44, 49, 51, 59, 65, 67, 76, 115, 123, 124, 141, 175, 184, 218, 247,
            259, 274, 275, 277, 297, 303, 310, 324, 351, 411, 415, 426, 430, 432, 462,
            514, 530, 535, 539, 549, 553, 562, 567, 589, 598, 602, 606, 614, 619, 627,
            629, 679, 725, 746, 765, 782, 797, 818, 830, 856, 866, 907, 932, 939, 943,
            945, 953, 968, 975, 979, 988, 1001, 1005, 1031, 1033, 1036, 1049, 1051,
            1053, 1054, 1088, 1090, 1113, 1125, 1145, 1157, 1194, 1206, 1233, 1235,
            1253, 1282, 1284, 1306, 1353, 1361, 1365, 1377, 1382, 1418, 1434, 1438,
            1476
            ]
    if args.dataset_name == 'PAMAP2':
        fs_base = 100
        window_T = 5.12
        l = [
            0, 3, 5, 9, 15, 17, 18, 22, 24, 25, 30, 31, 33, 39, 42, 45, 46, 55, 56, 57,
            63, 70, 72, 73, 76, 77, 78, 82, 84, 90, 93, 94, 101, 104, 108, 110, 114, 124,
            126, 131, 132, 137, 152, 155, 157, 165, 167, 172, 175, 176, 181, 184, 192,
            193, 194, 203, 211, 220, 222, 223, 225, 227, 229, 233, 238, 247, 248, 255,
            258, 268, 271, 277, 284, 287, 301, 307, 316, 321, 324, 332, 334, 335, 337,
            346, 361, 369, 378, 380, 383, 388, 389, 390, 391, 392, 395, 397, 398, 401,
            405, 406
            ]

    initialize_bmi160()
    print("BMI160 Initialized")
    auto_calibrate()


    results = read_configuration_return_results(l, window_T, window_len, fs_base, args.dataset_name, args.num_exits, args.proportions, args.th_combination, sensor_on=sensor_on,sensor_sleep=sensor_sleep)
    os.system("pkill -f 'python3 data_logger.py'")

    output_file = f'acc/{args.dataset_name}_accuracy_results.csv'
    header = ['t_start','t1', 't2', 't3', 't4', 'total', 'true_label', 'prediction', 'correctness', 'exit_taken', 'data%', "sensor_total_on_sec", "sensor_total_off_sec"]
    with open(output_file, "w", newline="") as f1:
        add_header(f1, header)
        write_content_to_file(f1, results, header)
                