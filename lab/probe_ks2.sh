#!/bin/bash
# ks2 环境探测
echo "=== 主机/账号 ==="
hostname; whoami; uname -a | cut -c1-120
echo "=== OS ==="
(cat /etc/os-release | head -2) 2>/dev/null
echo "=== CPU / 内存 ==="
nproc; free -g | head -3
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  nvidia-smi | grep -E "CUDA Version" | head -1
else
  echo "no nvidia-smi"
fi
echo "=== Python ==="
which python3 python 2>/dev/null; python3 -V 2>&1
echo "=== 关键包 ==="
python3 - <<'PY' 2>&1 | tail -8
import importlib
for m in ("torch","torchvision","numpy","PIL","cv2","scipy","tensorboard"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING")
try:
    import torch
    print("cuda_available:", torch.cuda.is_available(),
          "| device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0,
          "| cuda:", torch.version.cuda)
except Exception:
    pass
PY
echo "=== 磁盘 ==="
df -h ~ / 2>/dev/null | sort -u
echo "=== 家目录 ==="
ls -la ~ | head -15
echo "=== 已有环境/缓存 ==="
(conda env list 2>/dev/null || echo "no conda") | head -5
echo "=== 网络（能否访问外网 pip）==="
timeout 8 python3 -c "import urllib.request;print('pypi:', urllib.request.urlopen('https://pypi.tuna.tsinghua.edu.cn/simple/',timeout=6).status)" 2>&1 | tail -1
