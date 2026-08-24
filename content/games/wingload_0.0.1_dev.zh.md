---
title: "Wingload 0.0.1 开发手记：无头测试、移植 GDScript 与 Web 导出"
date: 2026-08-24
description: "AI agent 驱动的 Godot 开发流程实录：游戏内测试 harness、C# 全量移植 GDScript、导出管线与运行时验证，以及一份实战踩坑清单。"
type: docs
comments: false
---

[Wingload 0.0.1]({{< relref "wingload_demo_0.0.1" >}}) 的开发全程由 AI agent 驱动：代码是 agent 写的，但 agent 没法亲手玩游戏。这条约束决定了整套工作流的形态——游戏必须自带"可被脚本驾驶"的测试钩子，否则 agent 只能说"代码写完了"，不能说"游戏能玩"。这篇文章把 0.0.1 开发中的三条主线整理成文：**无头测试 harness**、**C# → GDScript 全量移植**、**导出与运行时验证**，最后附一份实战踩坑清单。

## 无头测试 harness：唯一的回归网

核心做法是在主场景脚本里实现两个由环境变量门控的测试钩子，**留在正式发布的代码里**（门控后开销为零）：

- `WINGLOAD_SMOKE=1` —— 帧脚本化（frame-scripted）自动测试：按帧号触发动作，包括模拟按键、调用 debug 接口掉金币/掉机舱、直接 spawn 敌人和 boss、走完整改装流程。每步打印 `[SMOKE] PASS: ...` / `[SMOKE] FAIL: ...`，结束时打印汇总并 `get_tree().quit(0)`。0.0.1 的 smoke 有 28 项检查。
- `WINGLOAD_SHOT=<dir>` —— 运行到指定帧时 `get_viewport().get_texture().get_image().save_png(...)` 截图后退出。注意截图**必须窗口运行**，headless 模式没有渲染输出，产出的只会是空图。

几个设计要点：

1. **确定性优先**。随机掉落这类不确定因素要用测试专用路径绕过——smoke 里禁用环境 spawn，用 debug grant 代替随机掉落，否则测试会 flaky。
2. **每个检查一行 PASS/FAIL，带上终值断言**。比如金币数逐步对账：`gold 500 -> 505, expect 505`。agent 读 log 就能直接定位断在哪一步。
3. **胜负路径都要测**。胜利 overlay、失败 overlay、暂停规则的拒绝路径（战斗中按 Tab 打开改装被拒）都是独立检查项。

在这个 harness 之上，验证按阶梯逐级上升，**每级都要亲眼看输出，不能只看 exit code**：

```bash
# 1. 编译（C# 项目；GDScript 项目改跑下一行的 import 检查）
dotnet build

# 2. 编辑器 headless import —— 抓 script error / 资源导入错误
godot --headless --editor --quit --path .

# 3. 冒烟测试 —— 逻辑验证主力
WINGLOAD_SMOKE=1 godot --headless --path .

# 4. 截图 —— 视觉验证，agent 亲自看图
WINGLOAD_SHOT=/tmp/shots godot --path .

# 5. soak run —— 长时间无脚本运行，抓 smoke 覆盖不到的运行时错误
timeout 420 godot --headless --path .
```

两个容易误判的地方：smoke 的 exit code 只代表"跑完了"，通过标准应该是"exit 0 且 grep 出 N 个 PASS 且 0 个 FAIL"；soak run 无人操控，战机在 wave 4 左右被打爆是正常现象，不算错误。另外退出时的 `resources still in use` 警告多为退出瞬间仍在播放的 ogg 音效，benign，不影响 exit code。

## 导出管线

### Export templates 与 CLI 导出

Linux 发行版打包的 Godot（如 Arch 的 `godot`/`godot-mono`）**不包含** export templates，需要手动下载 tpz 解压，按 editor 版本放到对应目录（注意 `.mono` 后缀）：

- 标准版：`~/.local/share/godot/export_templates/4.7.2.stable/`
- mono 版：`~/.local/share/godot/export_templates/4.7.2.stable.mono/`

**mono 的 tpz 里没有 web 模板**（因为 C# 根本不能导出 web，下文详述），web 导出要用标准版 tpz；两套模板可以共存于不同版本目录。下载完记得用 `file` + `unzip -t` 验证完整性——下载源出错时 `curl` 可能照常保存 HTML 错误页并返回 0，给你一个几百字节的"假成功"tpz（加 `-f` 参数可以避免）。

CLI 导出本身很简单：

```bash
godot --headless --path . --export-release <preset名> <输出路径>
```

preset 定义在 `export_presets.cfg`（纯文本，可手写，缺省选项自动补默认值）；输出**目录**要先自己 `mkdir -p`，导出器不会建。C# 项目导出前确保 `dotnet build` 通过，导出器会把构建产物打进 pck；Linux preset 用 `binary_format/embed_pck=true` 可产出单文件可执行。

### 让导出物自己跑冒烟

导出成功 ≠ 导出物能跑。因为 smoke 钩子是 env var 门控、随包导出的，可以直接验证导出二进制：

```bash
WINGLOAD_SMOKE=1 ./build/linux/wingload.x86_64 --headless   # 28/28 PASS
```

这比"导出成功"强得多——它证明脚本程序集、资源、data JSON 全部正确打包。

## C# → GDScript 全量移植

### 动机与决策

0.0.1 最初是 C# 项目，但 Godot 4 的 C# 无法导出 Web——mono 编辑器会直接拒绝并明示：

> Exporting to Web is currently not supported in Godot 4 when using C#/.NET.

社区有过原型（godot#106125），但到 4.7.2 仍未落地，**没有官方时间表**。要 web 就只有全量移植 GDScript 一条路。移植还有个附带收益：去掉 mono 依赖后导出体积小一截，免去 `dotnet build` 步骤，编辑器迭代更快。

一个关键决策：**不要双语共存**。场景文件只引用一份脚本，双语共存等于双份维护加行为漂移。要么不移植，要么删干净。

### 移植顺序：先保住回归网

1. **先移植测试 harness**（Main 里的 smoke/shot 钩子，469 行、25 个帧分支）——它是移植过程中唯一的安全网。早移植，先把骨架跑绿。
2. 纯逻辑类（数据记录、catalog、规则引擎）——基本 1:1 直译。
3. 节点层（场景树/UI/输入/音效）——API 1:1 对应，但信号重连量最大。
4. 场景文件改写 → 清理 → 全量验证。

### 语言映射表

| C# | GDScript |
|---|---|
| `event Action X` / `event Action<T>` | `signal x` / `signal x(arg)`；**每个订阅点都要手动重连**，漏接是静默 no-op，编译器不救你 |
| interface（如 `IEnemyHitTarget`） | duck typing：`node.has_method("take_hit")`；注意保留原来的类型分派**顺序** |
| static class / static field | `class_name` + `static var`/`static func`（Godot 4 支持静态成员），或 autoload |
| record / record struct | 小 class 或 Dictionary |
| `ref`/`out` 参数 | 返回 Array/Dictionary |
| LINQ（First/Any/Where） | 循环 / `filter()` / `any()` / `all()` |
| 类型 pattern switch | if 链 + `is` / `has_method`，顺序敏感的分支要逐个核对 |
| `[Export]` | `@export` |
| `async Task` + `ToSignal(...)` | `await signal_name` |
| enum | `enum Name { A, B }`（语法基本一样） |
| JSON（JsonDocument 等） | `JSON.parse_string()` + Dictionary——GDScript 版通常**更短** |
| 一文件多 class | 拆文件或 inner class；注意 tscn 挂的是哪个类 |
| Dictionary 以节点对象为 key | 可以，但节点重建/free 后要小心悬垂 key |

### 场景与工程文件

- 所有 .tscn 里的 `ext_resource path="res://scripts/**.cs"` 逐条改成 .gd 路径（uid 不用管，editor import 时自动重生成）。
- 删除清单：所有 `.cs`、所有 `.cs.uid`、`.csproj`、`.sln`、`.godot/mono/`。
- `project.godot` 通常**零改动**。
- 移植后跑一次 `godot --headless --editor --quit --path .` 让编辑器重建 import 缓存，读输出抓 script error。

### 验证与实测结果

验证顺序：editor headless import 无 script error → smoke 全部 PASS（与 C# 版同一套检查）→ 截图对比 C# 版旧图（desktop + mobile 两套）→ 60s+ 非 smoke 的 headless soak → 最终 grep 确认无 .cs 残留、无 .tscn 引用 .cs。

实测数据：4846 行 C#（34 文件）→ 4435 行 GDScript（45 文件，C# 的多 class 文件被拆分）。GDScript 更短，JSON catalog 部分缩水最明显。全部 class 用与 C# 同名的 `class_name`，交叉引用 1:1 可读，移植审查省力。

对 agent 来说，这是一次大任务但**机械性为主**：没有线程、反射、泛型重活。真正的风险集中在两处——信号重连和 duck-typing 分派顺序。验收标准永远以 smoke 全绿为准，不以"文件都翻完了"为准。

移植过程中还踩了两个新坑：

- **`Array.filter()` 返回无类型 Array**，不能赋回 `Array[Enemy]` 这样的 typed array——smoke 没覆盖到，60s soak 才暴露。改成倒序 `remove_at` 循环解决。教训：typed array 的赋值兼容性检查在运行时才炸，soak 不可省。
- `RandomNumberGenerator(seed=42)` 与 C# `System.Random(42)` 序列不同——确定性仍在，但"和旧版逐像素一致"做不到（星空背景位置变了，无害）。

移植完成后 Web 导出立即打通，wasm 产物 38 MB。

## Web 导出物的运行时验证

Web 导出成功同样 ≠ 能跑。wasm 在浏览器里的实际运行情况需要专门验证：

1. **本地 HTTP 服务必须带 COOP/COEP 头**——线程版 wasm 要求 cross-origin isolation：

   ```python
   class H(http.server.SimpleHTTPRequestHandler):
       def end_headers(self):
           self.send_header("Cross-Origin-Opener-Policy", "same-origin")
           self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
           super().end_headers()
   ```

2. **headless Chromium 截图不能用 `--virtual-time-budget`**——它对 wasm 启动无效，截图永远停在加载页。正确做法是走 CDP 真实等待：挂 `--remote-debugging-port`，`Page.navigate` 后真实 sleep 40s+，再 `Page.captureScreenshot`；同时订阅 `Runtime.consoleAPICalled` 收游戏内 print 日志——日志里出现 `[WAVES] wave N starting` 就直接证明游戏在跑。Node ≥ 21 内置 WebSocket 客户端，写个几十行的脚本即可驱动 CDP。软件渲染环境加 `--enable-unsafe-swiftshader --disable-gpu`。

3. **静态托管要用 nothreads 变体**。线程版 wasm 要求服务端发 COOP/COEP 头，GitHub Pages 这类静态托管做不到。在 `export_presets.cfg` 里复制 Web preset、改 `variant/thread_support=false` 做成独立 preset，产物用不带任何特殊头的 `python3 -m http.server` 就能正常启动（控制台显示 "single-threaded" 构建）。代价是潜在性能下降，所以做成可选 preset 而非替换默认。

## 踩坑清单

最后是一份实战坑清单，每条都真实卡过人：

**物理与场景树**

- **"Can't change state while flushing queries"**：在 Area2D 的 `area_entered` 等物理回调里直接 `add_child()`（敌人死亡掉金币）会报错。解法：`call_deferred("add_child", node)`，或对 collision 属性用 `set_deferred()`。
- **pause 会冻结 Timer/Tween**：boss 死亡瞬间触发胜利 overlay（pause 整棵树），原本设计的"连环爆炸延时序列"被冻住。解法：关键反馈在 pause 前一次性播完，或给 overlay 节点设 `process_mode = ALWAYS`。
- **场景实例化的位置时序**：`add_child` 前设置 `position` 可能被 `_Ready`/布局覆盖，mobile 布局平移场景后尤其明显。解法：先 `add_child`，再设 `global_position`。

**坐标系**

- **出界判定要用局部坐标**：mobile 模式把整个战场容器平移 −360px，用 `global_position` 对 canvas 坐标系矩形做 despawn 判定，会导致敌人一出生就被销毁。统一改用容器局部 `position` 后，desktop（identity transform）行为不变。教训：**比较坐标前，先确认两边在同一个坐标系**。
- **质心重锚定会移动舰船 origin**：机舱损毁后围绕新质心重建机体，origin 会偏移，所有"在舰船前方 spawn 测试敌人"的硬编码坐标随之失效。测试/生成逻辑要从 ship 节点取实时位置，不要写死。

**渲染与 UI**

- **CanvasLayer（HUD）画在 canvas items 之上**：顶缘 40px 的 HUD 条会盖住游戏世界顶部的预警箭头。设计战场布局时要把 HUD 高度让出来。
- **Kenney 字体没有 CJK 字形**，中文直接变豆腐块。更深一层的坑：**SystemFont fallback 在 web 导出里完全无效**（浏览器里没有系统字体），桌面好好的中文到 web 全变豆腐。正确做法是打包子集化字体：用 fonttools 的 `pyftsubset` 从 Noto Sans CJK SC（OFL 许可）子集化出仅含游戏用到的 247 个 CJK 字符的 103KB otf，统一挂为 Kenney 字体的 fallback。新增中文字符后要重新生成子集。另外 ▮/▯ 这类块字符 Kenney 也没有，用 `#`/`-` 代替。

**窗口与平台**

- **tiling 窗口管理器会无视 `--resolution`**（如 Hyprland），窗口被强制 tile 成奇怪尺寸。对策：stretch 设 `canvas_items` + `expand`，布局任意尺寸自适应；需要精确截图尺寸时用 WM 的 IPC（如 hyprctl）把窗口 float + resize 后再截。
- **headless editor 不生成 C# 工程文件**：`godot-mono --headless --editor --quit` 不会自动建 `.csproj`/`.sln`（GUI 里加第一个 C# 脚本才会），手写一个几行的 csproj 即可，`Godot.NET.Sdk` 版本与引擎版本对齐。

**音效**

- **枪声必须节流**：多武器同时自动开火，每个都 `play()` 会糊成一片。按音效类型限频（≤4 次/秒）即可，不用对象池。

## 结语

0.0.1 的流程可以浓缩成一句话：**把测试钩子留在发布代码里，把验证阶梯变成肌肉记忆**。harness 是唯一的回归网，任何重构和移植先保证它绿；验收标准永远是 smoke 全绿加亲眼看过截图，而不是"代码写完了"。至于那次 C# → GDScript 移植，最大的感想是：机械性的大任务恰恰最适合 agent——只要回归网够硬，风险就只剩信号重连和分派顺序这两处需要人工复核的地方。
