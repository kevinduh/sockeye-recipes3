#!/usr/bin/env bash

# TODO: this script is in flux due to changes from MXNet to PyTorch
# Further testing needed

# Sets $device variable to --use-cpu if CPU or --use-device-id X if GPU
#
# The default is to infer whether to use CPU or GPU based on
# the absence or presence of mxnet-cu* library. 
# Optional 1st argument (DESIRED_DEVICE) can overide this auto-check.
#
# If GPU is desired, the script then picks the proper device id based 
# first on $CUDA_VISIBLE_DEVICES if set, and if not it will choose 
# a free GPU randomly based on nvidia-smi result
# All this assumes we use --disable-device-locking in Sockeye
#
# This script is meant to be sourced by train.sh, etc.,
# which depend on the $device and $devicelog variables set here.


USE_DEVICE=$1
if [[ -z $USE_DEVICE ]]; then
    USE_DEVICE="gpu"
fi


# 3. Set $device variable for Sockeye
if [ "$USE_DEVICE" == "cpu" ]; then
    # Use CPU
    device="--use-cpu"
    devicelog="get-device.sh: On $HOSTNAME Using CPU."
    
elif [ "$USE_DEVICE" == "gpu" ]; then
    # Use GPU
    gpuprocessPID=$(nvidia-smi -q -x | grep pid | sed -e 's/<pid>//g' -e 's/<\/pid>//g' -e 's/^[[:space:]]*//')
    [[ -n "$gpuprocessPID" ]] && gpuprocess=$(ps -up $gpuprocessPID) || gpuprocess=""
    nvsmi=`nvidia-smi`

    if [ $CUDA_VISIBLE_DEVICES ]; then
	# If CUDA_VISIBLE_DEVICES is already set, just return 0,1,...,maxid
	# Setting CUDA_VISIBLE_DEVICES to multiple devices enables multi-gpu jobs
	visible=(${CUDA_VISIBLE_DEVICES//,/ })
	maxid=$(expr "${#visible[@]}" - 1)
	visible_mapping=$(seq -s ' ' 0 $maxid)
	device="--device-id $visible_mapping"
	devicelog="get-device.sh: On $HOSTNAME CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES found, map to device-ids = $visible_mapping. $nvsmi $gpuprocess"
	
    else
	# Else, look for a single free GPU
	# Note, we assume single GPU use here for safety.
	# The following script picks a GPU with no utilization, 
	# but there is no guarantee that some other process has not reserved it.
	# Setting CUDA_VISIBLE_DEVICES is the recommended way.
	freegpu=( $( nvidia-smi --format=noheader,csv --query-gpu=index,utilization.gpu | grep ', 0 %' | awk -F ',' '{print $1}' ) )
	
	numfree=${#freegpu[@]}
	
	if [ $numfree -gt 0 ]; then
            # Randomly pick one gpu id to return
            pick=$(expr $RANDOM % $numfree )
            device="--device-ids ${freegpu[$pick]}"
            devicelog="get-device.sh: Picking device-id = ${freegpu[$pick]} from free GPUs: ${freegpu[@]}. $nvsmi $gpuprocess"
	else
            # No GPUs, default back to CPU
            device="--use-cpu"
            devicelog="get-device.sh: On $HOSTNAME Using CPU because no free GPUs. $nvsmi $gpuprocess"
	fi
    fi
else
    echo "Invalid device name specified; must be one of cpu or gpu"
    exit 1
fi

echo "$device"
echo "$devicelog"
echo 
