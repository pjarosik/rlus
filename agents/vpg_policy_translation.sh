#!/bin/bash
set -e

if [ -z "${PROJECT_PATH}" ]; then
    echo "PROJECT_PATH is unset/empty, set it to the path, where the project is placed. "
    exit 1
fi
EXP_DIR=$1

export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH
mkdir $1
cp vpg_policy_translation.py $EXP_DIR/vpg_policy_translation.py
cd $1
python3 vpg_policy_translation.py --exp_dir $EXP_DIR
cd -
