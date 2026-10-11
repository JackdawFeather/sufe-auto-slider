#!/bin/bash
# ks2 远端训练启动器（torch + A40 GPU）
cd /gemini/code/sufe-slider || exit 1
pkill -f "[t]rain2" 2>/dev/null
sleep 2
setsid nohup env SUFE_CHANNELS=24,32,48,64 SUFE_BS=256 \
  /root/miniconda3/bin/python -u train2_torch.py 600 > train_gpu.log 2>&1 < /dev/null &
sleep 10
echo "--- 进程 ---"
ps aux | grep "[t]rain2_torch" | head -2
echo "--- 日志 ---"
tail -8 train_gpu.log 2>/dev/null
