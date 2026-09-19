# 初試 Omarchy

> Omarchy 裝機後變更記錄與使用體驗

---

LLMS index: [llms.txt](/llms.txt)

---

## 安裝時的注意事項

- **關閉 Secure Boot**：部分主板/BIOS 需要先設置管理員密碼，才能把 Secure Boot 關掉。
- **關閉 Fast Boot**：避免啓動過程跳過設備初始化，導致引導或硬件識別問題。
- **選擇 UEFI 啓動**：確保安裝介質以 UEFI 模式啓動。

## 裝機後變更記錄

- **輸入法**：安裝 fcitx5 + Rime，啓用簡體輸入方案 `luna_pinyin_simp`；補充缺失的 OpenCC 繁轉簡字典；移除繁簡切換開關，強制簡體輸出。
- **編輯器與 AI 助手**：安裝 VS Code 與 Kimi CLI（`/home/pog/.kimi-code/bin/kimi`）。
- **把 Kimi 包成圖形化 App**：創建 `~/.local/share/applications/kimi.desktop` 與圖標，修正 systemd 圖形環境的 `PATH`，`.desktop` 的 `Exec` 使用絕對路徑，刷新桌面數據庫、圖標緩存與 Omarchy 菜單。
- **觸摸板**：開啓自然滾動（macOS 方向），在 `~/.config/hypr/input.lua` 設置 `natural_scroll = true`。
- **免密碼 sudo**：在 `/etc/sudoers.d/omarchy-pkg` 加入 `pog ALL=(ALL) NOPASSWD: /usr/bin/pacman`，讓 Kimi 在沒有 root 交互的情況下也能自行運行 `omarchy pkg add` 安裝軟件包。
- **小屏顯示優化**：全局關閉窗口透明，並把窗口內外縫隙設爲 `0`，保留 `1px` 邊框來辨認焦點。
- **終端字體**：試用了 Caskaydia、Iosevka Term、Monaspace、Victor Mono、Fantasque、Fira Code、Monofur、Comic Shanns 等多款 Nerd Font，最終選回 `ComicShannsMono Nerd Font`；同時新建了一個 font skill，方便以後查找、安裝和切換字體。
- **Codex 執行策略**：在 `~/.codex/config.toml` 設置 `approval_policy = "never"` 與 `sandbox_mode = "danger-full-access"`，讓新會話默認免審批並獲得完整本機訪問能力。
- **MacBook 觸控板手勢**：啓用三指拖移；三指左滑、右滑改爲立即切換到下一個、上一個 workspace，效果接近 `SUPER + TAB`，比內置滑動動畫更直接。

## 使用體驗

這臺喫灰了兩年、原本接近報廢的 Acer 非凡 S3，跑 Windows 時風扇聲音極大，換成 Omarchy 之後就正常了。

UI/UX 非常簡潔美觀，minimalist 本身就是一種美。

一切都基於文本、基於代碼，對 agent 非常友好。大多數配置修改和客製化問題，用 Kimi 都能幫忙搞定。

換到 2019/20 款 MacBook 上，Omarchy 的體驗更完整了：macOS 式觸摸板自然滾動加三指拖移，讓單手操作也很順手；三指橫滑切換 workspace 後，效率甚至比原來的快捷鍵更高。Retina 屏幕的細膩畫質也很適合 Linux 桌面和終端閱讀；鍵盤上方的 Touch Bar 也能正常工作。

![在這臺 Acer 非凡 S3 上跑着 Omarchy 與 Kimi](/img/omarchy820.jpg)
