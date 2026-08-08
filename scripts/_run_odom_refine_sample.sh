#!/bin/bash
# Launcher: RC-5 odometry-edge sampling, detached with log (WSL gotcha:
# complex inline quoting gets eaten; keep it in a file).
source /root/miniconda3/bin/activate rap-inference
cd /mnt/e/githouse/RAP-SLAM
python -c "import ast; ast.parse(open('scripts/eval_sparse_odom_refine.py').read()); print('syntax OK')" || exit 1
mkdir -p output_odom_refine
setsid python scripts/eval_sparse_odom_refine.py --phase sample \
  > output_odom_refine/sample.log 2>&1 < /dev/null &
sleep 3
echo launched
