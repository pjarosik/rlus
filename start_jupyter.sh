#!/bin/bash
# Should be called from /rlus directory.
export PYTHONPATH=$(pwd):$PYTHONPATH
echo "Working on $PYTHONPATH..."
jupyter notebook
