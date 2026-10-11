"""发布 GitHub Release：创建 release 并上传 dist 下的 zip/crx 附件。

用法：python release.py [版本号]   例：python release.py 2.0.0
Token 优先取 $GITHUB_TOKEN，否则从 git credential fill（GCM）读取，不落盘、不打印。
"""
import json
import os
import subprocess
import sys
import urllib.request
from urllib.error import HTTPError

OWNER, REPO = "haohaoo3o", "sufe-auto-slider"
PROXY = os.environ.get("SUFE_PROXY", "http://127.0.0.1:7890")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.environ.get("SUFE_DIST") or next(
    (p for p in (os.path.join(os.path.dirname(ROOT), "dist"), os.path.join(ROOT, "dist"))
     if os.path.isdir(p)), os.path.join(ROOT, "dist"))

BODY = """## 上财自动滑块 3.0

滑块缺口识别升级为**整图输入的 CNN**，并把这轮改进的训练管线一并开源；模型在匡时一号算力平台的 A40 GPU 上训练。

### 亮点

- **整图 CNN 定位缺口**：背景图不做裁剪，整体 area-resize 到 256×128 → 4×[Conv3x3+ReLU → 高度方向 MaxPool] → 1×1 头 → 256 列 softmax → soft-argmax（4.9 万参数，纯 JS 前向、零依赖）
- **直接回归「可通过窗口中心」**：线上实测窗口恒为 `缺口左缘-5 … 左缘-1`（中心 = 左缘-3、半径 ±2），预测值四舍五入即命中，两侧余量对称
- **三级兜底**：CNN → CNN ±1/±2 邻域 → 经典「平坦纯白连通块」分割 → 历史盲试；失败不作废 token，同一 token 连续校验
- **交互改进**（合并社区 v2.1.0 思路）：倒计时缩短为 **2 秒**；只认点击/打字取消（浏览器翻译弹窗焦点变化不再误取消）；弹窗「登录」支持免刷新注入；密码显示/隐藏；首探命中率统计；`SSO10001` 学号不存在提示
- **训练管线开源**（`lab/`）：整图采集 → 标签重建 → GPU 训练（含缺口改色 + 亮块干扰增强）→ JS/训练端前向一致性 → 在线通过率验证

### 实测

| 场景 | 结果 |
|---|---|
| 留出集 76 张（离线） | MAE **0.47px**、≤2px **99%**、max 4.6px |
| JS ↔ 训练端前向一致 | 0.23px（取整零分歧） |
| 线上真实接口（新验证码） | 12/12 **首探测通过** |
| 真实浏览器内（登录页） | 5/5 首探测通过 |

### 安装

1. 下载下方 `sufe-auto-slider-v3.0.0.zip` 解压
2. Chrome → `chrome://extensions/` → 开启「开发者模式」→「加载已解压的扩展程序」→ 选择解压出的 `sufe-auto-slider` 文件夹（或把 `.crx` 拖进扩展页）
3. 点击工具栏扩展图标，填写学号、密码

> 仅供本人账号的个人自动化使用；扩展只向 login.sufe.edu.cn 提交数据，密码仅存本机 `chrome.storage.local`。

**升级说明**：与 v1.0.0 / v2.0.0 使用同一签名密钥（扩展 ID `nebckpodbmebhnfaljnmajlpcmbmolfc` 不变），可直接覆盖升级。
"""


def token():
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line[len("password="):]
    sys.exit("未找到 GitHub 凭据（GITHUB_TOKEN 或 git credential）")


def api(method, url, data=None, token_=None, ctype="application/json"):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token_)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "sufe-release-script")
    if data is not None:
        req.add_header("Content-Type", ctype)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY, "http": PROXY}))
    with op.open(req, timeout=120) as r:
        return json.loads(r.read().decode() or "{}")


def main():
    ver = sys.argv[1] if sys.argv[1:] else "2.0.0"
    tag = "v" + ver
    tk = token()
    rel = api("POST", f"https://api.github.com/repos/{OWNER}/{REPO}/releases", json.dumps({
        "tag_name": tag, "name": f"{tag} · CNN 缺口识别", "body": BODY,
        "draft": False, "prerelease": False,
    }).encode(), tk)
    print("release:", rel.get("html_url") or rel)
    rid = rel["id"]
    for name in (f"sufe-auto-slider-v{ver}.zip", f"sufe-auto-slider-v{ver}.crx"):
        path = os.path.join(DIST, name)
        if not os.path.exists(path):
            print("!! 缺少附件，跳过：", path)
            continue
        raw = open(path, "rb").read()
        ct = "application/zip" if name.endswith(".zip") else "application/x-chrome-extension"
        try:
            a = api("POST",
                    f"https://uploads.github.com/repos/{OWNER}/{REPO}/releases/{rid}/assets?name={name}",
                    raw, tk, ct)
            print("asset:", a.get("name"), a.get("size"), "bytes")
        except HTTPError as e:
            print("!! 上传失败", name, e.code, e.read()[:300])


if __name__ == "__main__":
    main()
