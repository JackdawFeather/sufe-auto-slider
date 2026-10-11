#!/bin/bash
# ks2 远端训练启动器（由本地 Qoder 调用；机器：48C/881G，无 GPU）
cd /gemini/code/sufe-slider || exit 1
pkill -f "[t]rain2.py" 2>/dev/null
sleep 2
setsid nohup env OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 \
  SUFE_CHANNELS=24,32,48,64 SUFE_BS=96 SUFE_AUG=8 SUFE_LOG_EVERY=5 \
  /root/miniconda3/bin/python -u train2.py 60 > train_remote.log 2>&1 < /dev/null &
sleep 6
echo "--- 进程 ---"
ps aux | grep "[t]rain2.py" | head -2
echo "--- 日志头 ---"
tail -5 train_remote.log 2>/dev/null
