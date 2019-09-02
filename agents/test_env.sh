#!/bin/bash
set -e
TEST_METHOD=$1

if [ -z "${PROJECT_PATH}" ]; then
    echo "PROJECT_PATH is uneset/empty, set it to path, where the project is placed. "
    exit 1
fi

export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH
rm -rf $TEST_METHOD && mkdir $TEST_METHOD
cp test_env.py $TEST_METHOD/test_env_bck.py
python3 test_env.py $TEST_METHOD
