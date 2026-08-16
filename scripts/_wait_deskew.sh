#!/usr/bin/env bash
# Poll helper: wait up to ~280 s for the deskew batch to finish, then print status.
LOG=/mnt/e/githouse/RAP-SLAM/output_deskew/batch.log
for i in $(seq 1 27); do
    sleep 10
    if grep -q ALL_DONE "$LOG" 2>/dev/null; then break; fi
done
cat "$LOG" 2>/dev/null
pgrep -af 'run_pipeline' | grep -v pgrep || true
