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

BODY = """## 上财自动滑块 2.0

滑块缺口识别从「盲试候选 X」升级为 **CNN 缺口识别**，实测一次通过。

### 亮点

- **CNN 定位缺口**：轻量全卷积网络（conv3×3 → ReLU → MaxPool → conv3×3 → 列池化 → 1×1 conv，约 4.4k 参数，纯 JS 前向、零依赖），直接从背景图算出缺口横坐标
- **三级兜底**：CNN → 经典分割（平坦高亮连通块）→ ±1/±2 邻域 → 盲试，同一 token 内连续尝试
- **闭环自检**：访问 `login.sufe.edu.cn/login/#sufe-test-slider` 只跑滑块验证，不提交账号密码
- **训练管线开源**：`lab/` 含 采集→精确标注→训练→导出 JS 权重→JS/Python 前向一致性校验→在线通过率验证 全流程

### 实测

| 场景 | 结果 |
|---|---|
| 在线闭环（15 次新验证码） | 15/15 **一次通过**，误差 ≤1px |
| 真实浏览器内（5 次） | 5/5 通过（4 次首候选即 CNN，1 次分割兜底） |
| JS ↔ Python 前向一致性 | 逐样本一致（最大偏差 <2e-5） |

### 安装

1. 下载下方 `sufe-auto-slider-v2.0.0.zip` 解压
2. Chrome → `chrome://extensions/` → 开启「开发者模式」→「加载已解压的扩展程序」→ 选择解压出的 `sufe-auto-slider` 文件夹（或把 `.crx` 拖进扩展页）
3. 点击工具栏扩展图标，填写学号、密码

> 仅供本人账号的个人自动化使用；扩展只向 login.sufe.edu.cn 提交数据，密码仅存本机 `chrome.storage.local`。

**升级说明**：与 v1.0.0 使用同一签名密钥（扩展 ID `nebckpodbmebhnfaljnmajlpcmbmolfc` 不变），可覆盖升级。
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
