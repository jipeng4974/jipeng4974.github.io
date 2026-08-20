+++
title = "初试 Omarchy"
date = "2026-08-20"
tags = ["Linux"]
description = "Omarchy 装机后变更记录与使用体验"
showFullContent = false
+++

## 安装时的注意事项

- **关闭 Secure Boot**：部分主板/BIOS 需要先设置管理员密码，才能把 Secure Boot 关掉。
- **关闭 Fast Boot**：避免启动过程跳过设备初始化，导致引导或硬件识别问题。
- **选择 UEFI 启动**：确保安装介质以 UEFI 模式启动。

## 装机后变更记录

- **输入法**：安装 fcitx5 + Rime，启用简体输入方案 `luna_pinyin_simp`；补充缺失的 OpenCC 繁转简字典；移除繁简切换开关，强制简体输出。
- **编辑器与 AI 助手**：安装 VS Code 与 Kimi CLI（`/home/pog/.kimi-code/bin/kimi`）。
- **把 Kimi 包成图形化 App**：创建 `~/.local/share/applications/kimi.desktop` 与图标，修正 systemd 图形环境的 `PATH`，`.desktop` 的 `Exec` 使用绝对路径，刷新桌面数据库、图标缓存与 Omarchy 菜单。
- **触摸板**：开启自然滚动（macOS 方向），在 `~/.config/hypr/input.lua` 设置 `natural_scroll = true`。
- **免密码 sudo**：在 `/etc/sudoers.d/omarchy-pkg` 加入 `pog ALL=(ALL) NOPASSWD: /usr/bin/pacman`，让 Kimi 在没有 root 交互的情况下也能自行运行 `omarchy pkg add` 安装软件包。

## 使用体验

这台吃灰了两年、原本接近报废的 Acer 非凡 S3，跑 Windows 时风扇声音极大，换成 Omarchy 之后就正常了。

UI/UX 非常简洁美观，minimalist 本身就是一种美。

一切都基于文本、基于代码，对 agent 非常友好。大多数配置修改和客制化问题，用 Kimi 都能帮忙搞定。

![在这台 Acer 非凡 S3 上跑着 Omarchy 与 Kimi](/img/omarchy820.jpg)
