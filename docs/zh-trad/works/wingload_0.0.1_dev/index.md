# Wingload 0.0.1 開發手記：無頭測試、移植 GDScript 與 Web 導出

> AI agent 驅動的 Godot 開發流程實錄：遊戲內測試 harness、C# 全量移植 GDScript、導出管線與運行時驗證，以及一份實戰踩坑清單。

---

LLMS index: [llms.txt](/llms.txt)

---

[Wingload 0.0.1](/zh-trad/works/wingload_demo_0.0.1/) 的開發全程由 AI agent 驅動：代碼是 agent 寫的，但 agent 沒法親手玩遊戲。這條約束決定了整套工作流的形態——遊戲必須自帶"可被腳本駕駛"的測試鉤子，否則 agent 只能說"代碼寫完了"，不能說"遊戲能玩"。這篇文章把 0.0.1 開發中的三條主線整理成文：**無頭測試 harness**、**C# → GDScript 全量移植**、**導出與運行時驗證**，最後附一份實戰踩坑清單。

## 無頭測試 harness：唯一的迴歸網

核心做法是在主場景腳本里實現兩個由環境變量門控的測試鉤子，**留在正式發佈的代碼裏**（門控後開銷爲零）：

- `WINGLOAD_SMOKE=1` —— 幀腳本化（frame-scripted）自動測試：按幀號觸發動作，包括模擬按鍵、調用 debug 接口掉金幣/掉機艙、直接 spawn 敵人和 boss、走完整改裝流程。每步打印 `[SMOKE] PASS: ...` / `[SMOKE] FAIL: ...`，結束時打印彙總並 `get_tree().quit(0)`。0.0.1 的 smoke 有 28 項檢查。
- `WINGLOAD_SHOT=<dir>` —— 運行到指定幀時 `get_viewport().get_texture().get_image().save_png(...)` 截圖後退出。注意截圖**必須窗口運行**，headless 模式沒有渲染輸出，產出的只會是空圖。

幾個設計要點：

1. **確定性優先**。隨機掉落這類不確定因素要用測試專用路徑繞過——smoke 裏禁用環境 spawn，用 debug grant 代替隨機掉落，否則測試會 flaky。
2. **每個檢查一行 PASS/FAIL，帶上終值斷言**。比如金幣數逐步對賬：`gold 500 -> 505, expect 505`。agent 讀 log 就能直接定位斷在哪一步。
3. **勝負路徑都要測**。勝利 overlay、失敗 overlay、暫停規則的拒絕路徑（戰鬥中按 Tab 打開改裝被拒）都是獨立檢查項。

在這個 harness 之上，驗證按階梯逐級上升，**每級都要親眼看輸出，不能只看 exit code**：

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

兩個容易誤判的地方：smoke 的 exit code 只代表"跑完了"，通過標準應該是"exit 0 且 grep 出 N 個 PASS 且 0 個 FAIL"；soak run 無人操控，戰機在 wave 4 左右被打爆是正常現象，不算錯誤。另外退出時的 `resources still in use` 警告多爲退出瞬間仍在播放的 ogg 音效，benign，不影響 exit code。

## 導出管線

### Export templates 與 CLI 導出

Linux 發行版打包的 Godot（如 Arch 的 `godot`/`godot-mono`）**不包含** export templates，需要手動下載 tpz 解壓，按 editor 版本放到對應目錄（注意 `.mono` 後綴）：

- 標準版：`~/.local/share/godot/export_templates/4.7.2.stable/`
- mono 版：`~/.local/share/godot/export_templates/4.7.2.stable.mono/`

**mono 的 tpz 裏沒有 web 模板**（因爲 C# 根本不能導出 web，下文詳述），web 導出要用標準版 tpz；兩套模板可以共存於不同版本目錄。下載完記得用 `file` + `unzip -t` 驗證完整性——下載源出錯時 `curl` 可能照常保存 HTML 錯誤頁並返回 0，給你一個幾百字節的"假成功"tpz（加 `-f` 參數可以避免）。

CLI 導出本身很簡單：

```bash
godot --headless --path . --export-release <preset名> <输出路径>
```

preset 定義在 `export_presets.cfg`（純文本，可手寫，缺省選項自動補默認值）；輸出**目錄**要先自己 `mkdir -p`，導出器不會建。C# 項目導出前確保 `dotnet build` 通過，導出器會把構建產物打進 pck；Linux preset 用 `binary_format/embed_pck=true` 可產出單文件可執行。

### 讓導出物自己跑冒煙

導出成功 ≠ 導出物能跑。因爲 smoke 鉤子是 env var 門控、隨包導出的，可以直接驗證導出二進制：

```bash
WINGLOAD_SMOKE=1 ./build/linux/wingload.x86_64 --headless   # 28/28 PASS
```

這比"導出成功"強得多——它證明腳本程序集、資源、data JSON 全部正確打包。

## C# → GDScript 全量移植

### 動機與決策

0.0.1 最初是 C# 項目，但 Godot 4 的 C# 無法導出 Web——mono 編輯器會直接拒絕並明示：

> Exporting to Web is currently not supported in Godot 4 when using C#/.NET.

社區有過原型（godot#106125），但到 4.7.2 仍未落地，**沒有官方時間表**。要 web 就只有全量移植 GDScript 一條路。移植還有個附帶收益：去掉 mono 依賴後導出體積小一截，免去 `dotnet build` 步驟，編輯器迭代更快。

一個關鍵決策：**不要雙語共存**。場景文件只引用一份腳本，雙語共存等於雙份維護加行爲漂移。要麼不移植，要麼刪乾淨。

### 移植順序：先保住迴歸網

1. **先移植測試 harness**（Main 裏的 smoke/shot 鉤子，469 行、25 個幀分支）——它是移植過程中唯一的安全網。早移植，先把骨架跑綠。
2. 純邏輯類（數據記錄、catalog、規則引擎）——基本 1:1 直譯。
3. 節點層（場景樹/UI/輸入/音效）——API 1:1 對應，但信號重連量最大。
4. 場景文件改寫 → 清理 → 全量驗證。

### 語言映射表

| C# | GDScript |
|---|---|
| `event Action X` / `event Action<T>` | `signal x` / `signal x(arg)`；**每個訂閱點都要手動重連**，漏接是靜默 no-op，編譯器不救你 |
| interface（如 `IEnemyHitTarget`） | duck typing：`node.has_method("take_hit")`；注意保留原來的類型分派**順序** |
| static class / static field | `class_name` + `static var`/`static func`（Godot 4 支持靜態成員），或 autoload |
| record / record struct | 小 class 或 Dictionary |
| `ref`/`out` 參數 | 返回 Array/Dictionary |
| LINQ（First/Any/Where） | 循環 / `filter()` / `any()` / `all()` |
| 類型 pattern switch | if 鏈 + `is` / `has_method`，順序敏感的分支要逐個覈對 |
| `[Export]` | `@export` |
| `async Task` + `ToSignal(...)` | `await signal_name` |
| enum | `enum Name { A, B }`（語法基本一樣） |
| JSON（JsonDocument 等） | `JSON.parse_string()` + Dictionary——GDScript 版通常**更短** |
| 一文件多 class | 拆文件或 inner class；注意 tscn 掛的是哪個類 |
| Dictionary 以節點對象爲 key | 可以，但節點重建/free 後要小心懸垂 key |

### 場景與工程文件

- 所有 .tscn 裏的 `ext_resource path="res://scripts/**.cs"` 逐條改成 .gd 路徑（uid 不用管，editor import 時自動重生成）。
- 刪除清單：所有 `.cs`、所有 `.cs.uid`、`.csproj`、`.sln`、`.godot/mono/`。
- `project.godot` 通常**零改動**。
- 移植後跑一次 `godot --headless --editor --quit --path .` 讓編輯器重建 import 緩存，讀輸出抓 script error。

### 驗證與實測結果

驗證順序：editor headless import 無 script error → smoke 全部 PASS（與 C# 版同一套檢查）→ 截圖對比 C# 版舊圖（desktop + mobile 兩套）→ 60s+ 非 smoke 的 headless soak → 最終 grep 確認無 .cs 殘留、無 .tscn 引用 .cs。

實測數據：4846 行 C#（34 文件）→ 4435 行 GDScript（45 文件，C# 的多 class 文件被拆分）。GDScript 更短，JSON catalog 部分縮水最明顯。全部 class 用與 C# 同名的 `class_name`，交叉引用 1:1 可讀，移植審查省力。

對 agent 來說，這是一次大任務但**機械性爲主**：沒有線程、反射、泛型重活。真正的風險集中在兩處——信號重連和 duck-typing 分派順序。驗收標準永遠以 smoke 全綠爲準，不以"文件都翻完了"爲準。

移植過程中還踩了兩個新坑：

- **`Array.filter()` 返回無類型 Array**，不能賦回 `Array[Enemy]` 這樣的 typed array——smoke 沒覆蓋到，60s soak 才暴露。改成倒序 `remove_at` 循環解決。教訓：typed array 的賦值兼容性檢查在運行時才炸，soak 不可省。
- `RandomNumberGenerator(seed=42)` 與 C# `System.Random(42)` 序列不同——確定性仍在，但"和舊版逐像素一致"做不到（星空背景位置變了，無害）。

移植完成後 Web 導出立即打通，wasm 產物 38 MB。

## Web 導出物的運行時驗證

Web 導出成功同樣 ≠ 能跑。wasm 在瀏覽器裏的實際運行情況需要專門驗證：

1. **本地 HTTP 服務必須帶 COOP/COEP 頭**——線程版 wasm 要求 cross-origin isolation：

   ```python
   class H(http.server.SimpleHTTPRequestHandler):
       def end_headers(self):
           self.send_header("Cross-Origin-Opener-Policy", "same-origin")
           self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
           super().end_headers()
   ```

2. **headless Chromium 截圖不能用 `--virtual-time-budget`**——它對 wasm 啓動無效，截圖永遠停在加載頁。正確做法是走 CDP 真實等待：掛 `--remote-debugging-port`，`Page.navigate` 後真實 sleep 40s+，再 `Page.captureScreenshot`；同時訂閱 `Runtime.consoleAPICalled` 收遊戲內 print 日誌——日誌裏出現 `[WAVES] wave N starting` 就直接證明遊戲在跑。Node ≥ 21 內置 WebSocket 客戶端，寫個幾十行的腳本即可驅動 CDP。軟件渲染環境加 `--enable-unsafe-swiftshader --disable-gpu`。

3. **靜態託管要用 nothreads 變體**。線程版 wasm 要求服務端發 COOP/COEP 頭，GitHub Pages 這類靜態託管做不到。在 `export_presets.cfg` 裏複製 Web preset、改 `variant/thread_support=false` 做成獨立 preset，產物用不帶任何特殊頭的 `python3 -m http.server` 就能正常啓動（控制台顯示 "single-threaded" 構建）。代價是潛在性能下降，所以做成可選 preset 而非替換默認。

## 踩坑清單

最後是一份實戰坑清單，每條都真實卡過人：

**物理與場景樹**

- **"Can't change state while flushing queries"**：在 Area2D 的 `area_entered` 等物理回調裏直接 `add_child()`（敵人死亡掉金幣）會報錯。解法：`call_deferred("add_child", node)`，或對 collision 屬性用 `set_deferred()`。
- **pause 會凍結 Timer/Tween**：boss 死亡瞬間觸發勝利 overlay（pause 整棵樹），原本設計的"連環爆炸延時序列"被凍住。解法：關鍵反饋在 pause 前一次性播完，或給 overlay 節點設 `process_mode = ALWAYS`。
- **場景實例化的位置時序**：`add_child` 前設置 `position` 可能被 `_Ready`/佈局覆蓋，mobile 佈局平移場景後尤其明顯。解法：先 `add_child`，再設 `global_position`。

**座標系**

- **出界判定要用局部座標**：mobile 模式把整個戰場容器平移 −360px，用 `global_position` 對 canvas 座標系矩形做 despawn 判定，會導致敵人一出生就被銷燬。統一改用容器局部 `position` 後，desktop（identity transform）行爲不變。教訓：**比較座標前，先確認兩邊在同一個座標系**。
- **質心重錨定會移動艦船 origin**：機艙損毀後圍繞新質心重建機體，origin 會偏移，所有"在艦船前方 spawn 測試敵人"的硬編碼座標隨之失效。測試/生成邏輯要從 ship 節點取實時位置，不要寫死。

**渲染與 UI**

- **CanvasLayer（HUD）畫在 canvas items 之上**：頂緣 40px 的 HUD 條會蓋住遊戲世界頂部的預警箭頭。設計戰場佈局時要把 HUD 高度讓出來。
- **Kenney 字體沒有 CJK 字形**，中文直接變豆腐塊。更深一層的坑：**SystemFont fallback 在 web 導出裏完全無效**（瀏覽器裏沒有系統字體），桌面好好的中文到 web 全變豆腐。正確做法是打包子集化字體：用 fonttools 的 `pyftsubset` 從 Noto Sans CJK SC（OFL 許可）子集化出僅含遊戲用到的 247 個 CJK 字符的 103KB otf，統一掛爲 Kenney 字體的 fallback。新增中文字符後要重新生成子集。另外 ▮/▯ 這類塊字符 Kenney 也沒有，用 `#`/`-` 代替。

**窗口與平臺**

- **tiling 窗口管理器會無視 `--resolution`**（如 Hyprland），窗口被強制 tile 成奇怪尺寸。對策：stretch 設 `canvas_items` + `expand`，佈局任意尺寸自適應；需要精確截圖尺寸時用 WM 的 IPC（如 hyprctl）把窗口 float + resize 後再截。
- **headless editor 不生成 C# 工程文件**：`godot-mono --headless --editor --quit` 不會自動建 `.csproj`/`.sln`（GUI 里加第一個 C# 腳本纔會），手寫一個幾行的 csproj 即可，`Godot.NET.Sdk` 版本與引擎版本對齊。

**音效**

- **槍聲必須節流**：多武器同時自動開火，每個都 `play()` 會糊成一片。按音效類型限頻（≤4 次/秒）即可，不用對象池。

## 結語

0.0.1 的流程可以濃縮成一句話：**把測試鉤子留在發佈代碼裏，把驗證階梯變成肌肉記憶**。harness 是唯一的迴歸網，任何重構和移植先保證它綠；驗收標準永遠是 smoke 全綠加親眼看過截圖，而不是"代碼寫完了"。至於那次 C# → GDScript 移植，最大的感想是：機械性的大任務恰恰最適合 agent——只要迴歸網夠硬，風險就只剩信號重連和分派順序這兩處需要人工複覈的地方。
