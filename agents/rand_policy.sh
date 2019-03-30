#!/bin/bash
set -e

if [ -z "${PROJECT_PATH}" ]; then
    echo "PROJECT_PATH is uneset/empty, set it to path, where the project is placed. "
    exit 1
fi
export PYTHONPATH=${PROJECT_PATH}:$PYTHONPATH
python3 rand_policy.py "$@"
