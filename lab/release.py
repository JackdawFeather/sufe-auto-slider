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
# 默认直连；需要时用 SUFE_PROXY=http://127.0.0.1:7890 指定代理
PROXY = os.environ.get("SUFE_PROXY", "").strip()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.environ.get("SUFE_DIST") or next(
    (p for p in (os.path.join(os.path.dirname(ROOT), "dist"), os.path.join(ROOT, "dist"))
     if os.path.isdir(p)), os.path.join(ROOT, "dist"))

BODY = """## 上财自动滑块 4.0

**架构反转：缺口定位改为「经典分割严格优先 + CNN 兜底」**——多数验证码本地 2ms 内定位，CNN 只在渲染异常时才出场。

### 亮点

- **分割优先（快 515 倍 + 更准）**：缺口模板在图中恒为 ~96×93、面积 ~5590（382 张实测 99% 一致）；「形状/面积合规且全场唯一」的平坦纯白连通块即可信 → 本地 **~2ms、误差 0.000px**（对比整图 CNN：0.46px / 1.76s）
- **懒执行候选流**：严格分割 → 宽松分割 → CNN（~1-2s，仅必要时）→ 盲试；同一 token 连续校验、失败不作废
- **CNN 兜底经过验证**：把缺口涂灰模拟渲染变化后，严格分割 12/12 正确拒绝，CNN 兜底 12/12 误差 ≤2px
- 保留 v3 的整图 CNN（4.9 万参数，回归「可通过窗口中心」= 缺口左缘-3，线上实测窗口 `左缘-5 … 左缘-1`）
- 交互同 v3：2 秒倒计时、点击/打字取消、弹窗免刷新「登录」、密码显隐、首探命中率统计

### 实测（本版）

| 场景 | 结果 |
|---|---|
| 留出集 76 张 | 严格分割 **75/76 可信、误差 0.000px、2ms** |
| 真实浏览器内（登录页，8 轮） | **8/8 首探测通过**，全部走 2ms 分割路径，0 次触发 CNN |
| 渲染变化模拟（缺口涂灰，12 张） | 分割全部正确拒绝 → CNN 兜底 12/12 误差 ≤2px |
| v3 基线（供对照） | 留出集 CNN MAE 0.47px；在线 12/12；浏览器 5/5 |

### 安装

1. 下载下方 `sufe-auto-slider-v4.0.0.zip` 解压
2. Chrome → `chrome://extensions/` → 开启「开发者模式」→「加载已解压的扩展程序」→ 选择解压出的 `sufe-auto-slider` 文件夹（或把 `.crx` 拖进扩展页）
3. 点击工具栏扩展图标，填写学号、密码

> 仅供本人账号的个人自动化使用；扩展只向 login.sufe.edu.cn 提交数据，密码仅存本机 `chrome.storage.local`。

**升级说明**：与 v1/v2/v3 使用同一签名密钥（扩展 ID `nebckpodbmebhnfaljnmajlpcmbmolfc` 不变），可直接覆盖升级。

### 致谢

- **Yanzhe Lu**（上海财经大学，人工智能专业）：提出并实现了「**卷积 + 池化 CNN 定位缺口**」的整图输入方案（社区 v2.1.0 分支），并带来网格探测、免刷新注入等交互改进。本项目的整图模型与后续架构演进均受其启发，特此致谢。
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
    op = urllib.request.build_opener(
        urllib.request.ProxyHandler({"https": PROXY, "http": PROXY}) if PROXY else urllib.request.ProxyHandler({}))
    with op.open(req, timeout=120) as r:
        return json.loads(r.read().decode() or "{}")


def main():
    ver = sys.argv[1] if sys.argv[1:] else "4.0.0"
    tag = "v" + ver
    title = os.environ.get("SUFE_RELEASE_TITLE", "上财自动滑块").strip()
    tk = token()
    rel = api("POST", f"https://api.github.com/repos/{OWNER}/{REPO}/releases", json.dumps({
        "tag_name": tag, "name": f"{tag} · {title}", "body": BODY,
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
