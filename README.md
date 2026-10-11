# 上财自动滑块 (sufe-auto-slider)

检测到**上海财经大学统一认证登录页**（login.sufe.edu.cn）时，自动完成「滑块验证 + 登录」，成功后自动跳回原业务系统。

> 仅供本人账号的个人自动化使用。请控制请求频率，遵守学校相关规定；请勿用于他人账号或任何高频、批量场景。

## 功能特性

- 自动检测上财登录页（SPA 渲染完成后触发），右上角浮窗倒计时 4 秒后自动登录，表单打字即自动取消
- 滑块缺口识别采用**轻量全卷积 CNN**（conv → relu → maxpool → conv → 列池化 → 1x1 conv，约 4.4k 参数，纯 JS 前向、无依赖）：取背景图缺口纵带 → 统一宽度 512 → 2x2 下采样 → 输出 128 列 softmax 定位缺口横坐标；训练数据为真实验证码 + 服务端渲染缺口的精确分割标注
- CNN 之外保留两级兜底：经典「平坦高亮低饱和连通块」分割 → 盲试候选 X；候选梯队在同一 token 内依次尝试，通常**一次通过**
- 密码使用服务端下发的 RSA 公钥（PKCS#1 v1.5）加密后传输——浏览器 WebCrypto 不支持 v1.5 填充，扩展内置了一个无依赖的 BigInt 实现（`sufe-crypto.js`）
- 浮窗滚动显示每轮滑块尝试结果（含识别方式/CNN 置信度/分割坐标）；弹窗提供「测试滑块」（不提交账号）
- 闭环自检：访问 `login.sufe.edu.cn/login/#sufe-test-slider` 只跑滑块验证并在浮窗显示结果，不触碰账号密码
- 密码仅保存在本机 `chrome.storage.local`，不上传、不进浏览器同步；扩展只向 `login.sufe.edu.cn` 提交数据
- 登录成功后 20 秒内不重复触发（防跳转异常导致自动登录死循环）

## 安装（开发者模式加载）

1. 打开 Chrome，地址栏输入 `chrome://extensions/`
2. 右上角打开「**开发者模式**」
3. 点左上角「**加载已解压的扩展程序**」，选择本仓库克隆下来的 `sufe-auto-slider` 文件夹（或从 [Releases](https://github.com/haohaoo3o/sufe-auto-slider/releases) 下载最新 `sufe-auto-slider-vX.Y.Z.zip` 解压后选择该文件夹）
4. 点击工具栏扩展图标，填写学号、密码，勾选自动登录，点「保存」

日常无需任何操作：被业务系统跳到登录页时会自动处理并跳回。

## 工作原理

```
① GET  /api/v3/auth/queryAllValid        → RSA 公钥 + publicKeyId + loginTicket
② POST /api/v3/sliderCaptcha/init        → 背景图 sourceImage + 滑块图 newImage + Y + token
③ CNN  背景图纵带 → conv/pool 列分类      → 缺口横坐标 X（natual px，= 缺口左缘 - 2）
④ GET  /api/v3/sliderCaptcha/check       → X/Y/token 校验，通过得 vcode（失败 token 仍可用，可换候选重试）
⑤ POST /api/v3/auth/doLogin              → 学号 + RSA 加密密码 + vcode → 会话 Cookie
```

X 坐标约定来自登录页前端源码：`X = 拖动左偏移(显示px) × imgScale`，`imgScale = 背景图自然宽 / 显示宽`，
故 X 为**背景图自然像素**坐标；背景图宽度不固定（460–594），识别前统一为 512（裁剪/reflect 补齐，不缩放）。

登录成功后优先读取 `doLogin` 响应中的跳转地址，否则重载当前 URL 由 SSO 续跳（保留 service 参数）。

## 模型与训练（`lab/` 目录，随仓库提供，不进扩展包）

- `harvest.py` 拉取真实验证码并用「缺口=平坦纯白连通块」精确标注 X；`train.py` 训练 `cnn.py` 中的全卷积网络（numpy 手写 conv/pool 反向）
- `export_js.py` 导出 `sufe-cnn-weights.js`；`parity.js`（Node）校验 JS 前向与 Python 前向逐样本一致；`gradcheck.py` 校验反向；`live_test.py` 在线闭环验证识别→校验通过率

## 配置调整

轮数等常量集中在 `content.js` 顶部（`MAX_ROUNDS` 等）；盲试兜底候选为 `X_CANDIDATES`（仅在 CNN 与分割都失败时使用）。修改后在 `chrome://extensions` 中点扩展的「重新加载」即可生效。

## 排障

| 现象 | 处理 |
|---|---|
| 浮窗没出现 | 确认地址栏是 `login.sufe.edu.cn` 且页面已渲染出登录框（SPA 最长等待 20 秒） |
| 「学号或密码错误（SSO10002）」 | 在扩展弹窗中更正账号信息 |
| 浮窗显示方式=blind/多轮未通过 | 多为缺口样式变化或风控收紧期；用 `#sufe-test-slider` 自检页看 CNN/分割坐标是否合理 |
| 登录成功但页面没跳走 | 浮窗会提示，手动刷新或重新进入业务系统即可 |

## 更新日志

### v2.0.0 (2026-10-11)

- **缺口识别换用 CNN**：轻量全卷积网络（conv3×3 → ReLU → MaxPool → conv3×3 → 列 max-pool → 1×1 conv，128 列 softmax，约 4.4k 参数，纯 JS 前向零依赖）直接从背景图定位缺口横坐标，替代 v1 的「盲试候选 X」
- 坐标映射确认：`X = 缺口左缘 − 2`（背景图自然像素）；背景图尺寸不固定（宽 460–594），统一 512 后识别，并对不可能位置做列屏蔽
- 候选梯队改为：CNN → 经典分割兜底 → ±1/±2 邻域 → 盲试（同一 token 可连续校验，无需重开）
- 新增闭环自检入口：`login.sufe.edu.cn/login/#sufe-test-slider`（只跑滑块验证，不提交账号）
- 开源训练管线 `lab/`：采集 → 精确标注 → 训练 → 导出 JS 权重 → JS/Python 前向一致性校验 → 在线通过率验证
- 实测：在线 15/15、真实浏览器内 5/5 **一次通过**（首候选即 CNN 结果，误差 ≤1px）

### v1.0.0 (2026-10-08)

- 首个版本：检测登录页后自动完成滑块验证并登录（RSA 加密密码；滑块采用候选 X 盲试 + 多轮重试）

## 贡献与反馈

- 提交 Issue/PR 前请先在 `chrome://extensions` 中点「重新加载」复现问题
- **请勿**在 Issue、截图或日志中粘贴学号、密码、会话 Cookie 等凭据
- 登录页接口若改版（字段/路径变化），欢迎带上报错信息提 PR

## License

[MIT](LICENSE)
