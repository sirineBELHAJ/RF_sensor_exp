#!/bin/bash

run() {
  python3 data_logger.py "$1" &    
  LOGGER_PID=$!
  python3 Main.py --dataset_name "$1" "${@:2}"
  #wait $LOGGER_PID 2>/dev/null     
  echo "Finished $1."
}


run Shoaib --num_exits 4 --proportions 0.47 0.57 0.69 1 --th_combination 1.84 1.65 1.5
sleep 300

run EMGPhysical --num_exits 2 --proportions 0.25 1 --th_combination 1.35
sleep 300

run SelfRegulationSCP1 --num_exits 2 --proportions 0.25 1 --th_combination 0.69
sleep 300

run WESADchest --num_exits 3 --proportions 0.25 0.49 1 --th_combination 0.6 0.58
sleep 300

run Epilepsy --num_exits 2 --proportions 0.32 1 --th_combination 1.39
sleep 300

run PAMAP2 --num_exits 3 --proportions 0.38 0.53 1 --th_combination 1.6 1.08

echo "All done."





# python3 Main.py \
#     --dataset_name Shoaib \
#     --num_exits 4 \
#     --proportions 0.47 0.57 0.69 1 \
#     --th_combination 1.84 1.65 1.5 & python data_logger.py --dataset_name Shoaib

# sleep(400)

# python3 Main.py \
#     --dataset_name EMGPhysical \
#     --num_exits 2 \
#     --proportions 0.25 1 \
#     --th_combination 1.35

# python3 Main.py \
#     --dataset_name SelfRegulationSCP1 \
#     --num_exits 2 \
#     --proportions 0.25 1 \
#     --th_combination 0.69

# python3 Main.py \
#     --dataset_name WESADchest \
#     --num_exits 3 \
#     --proportions 0.25 0.49 1 \
#     --th_combination 0.6 0.58

# python3 Main.py \
#     --dataset_name Epilepsy \
#     --num_exits 2 \
#     --proportions 0.32 1 \
#     --th_combination 1.39

# python3 Main.py \
#     --dataset_name PAMAP2 \
#     --num_exits 3 \
#     --proportions 0.38 0.53 1 \
#     --th_combination 1.6 1.08