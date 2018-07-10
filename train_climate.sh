#!/bin/bash


export HDF5_USE_FILE_LOCKING=FALSE

module load tensorflow/intel-1.8.0-py27
module load h5py

export PYTHONPATH=/usr/common/software/h5py/2.7.1/lib/python2.7/site-packages:${PYTHONPATH}

#data
dataroot=/global/cscratch1/sd/tkurth/gb2018/tiramisu/segm_h5_v3_split #/project/projectdirs/dasrepo/gb2018/tiramisu/segm_h5_v3_split

python train_climate.py --data_dir=${dataroot} \
    --train_epochs=50 \
    --epochs_per_eval=1 \
    --batch_size=10 \
    --learning_rate_policy=poly \
    --data_dir=${dataroot} \
    --base_architecture=resnet_v2_101
    
