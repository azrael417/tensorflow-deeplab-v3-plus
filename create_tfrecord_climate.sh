#!/bin/bash

module load tensorflow/intel-1.8.0-py27

#data
dataroot=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_reformat #/project/projectdirs/dasrepo/gb2018/tiramisu/segm_h5_v3_reformat

python create_climate_tf_record.py --data_dir=${dataroot} --train_fraction=0.8 --output_path=${dataroot}/../segm_h5_v3_split
