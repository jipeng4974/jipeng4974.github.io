# First Impressions of Omarchy

> Post-install changes and hands-on experience with Omarchy

---

LLMS index: [llms.txt](/llms.txt)

---

## Installation notes

- **Disable Secure Boot**: some BIOSes require setting an administrator password before Secure Boot can be turned off.
- **Disable Fast Boot**: skipping device initialization during boot can cause boot or hardware-detection problems.
- **Boot in UEFI mode**: make sure the installation media boots in UEFI mode.

## Post-install changes

- **Input method**: installed fcitx5 + Rime and switched to the simplified-Chinese scheme `luna_pinyin_simp`; copied in the missing OpenCC traditional-to-simplified dictionaries; removed the traditional/simplified toggle so output is always simplified.
- **Editor and AI assistant**: installed VS Code and the Kimi CLI (`/home/pog/.kimi-code/bin/kimi`).
- **Wrapped Kimi as a graphical app**: created `~/.local/share/applications/kimi.desktop` plus an icon, fixed the systemd graphical-session `PATH`, used an absolute path in the `.desktop` `Exec`, and refreshed the desktop database, icon cache, and Omarchy menu.
- **Touchpad**: enabled natural (macOS-style) scrolling via `natural_scroll = true` in `~/.config/hypr/input.lua`.
- **Passwordless sudo for pacman**: added `pog ALL=(ALL) NOPASSWD: /usr/bin/pacman` to `/etc/sudoers.d/omarchy-pkg`, so Kimi can run `omarchy pkg add` to install packages without an interactive root prompt.
- **Small-screen display tuning**: disabled window opacity globally, removed inner and outer window gaps, and kept a `1px` border to identify the focused window.
- **Terminal fonts**: tried several Nerd Fonts, including Caskaydia, Iosevka Term, Monaspace, Victor Mono, Fantasque, Fira Code, Monofur, and Comic Shanns, then settled on `ComicShannsMono Nerd Font`. Also created a font skill to make it easy to find, install, and switch fonts later.
- **Codex execution policy**: set `approval_policy = "never"` and `sandbox_mode = "danger-full-access"` in `~/.codex/config.toml`, so new sessions run without approval prompts and with full local access by default.
- **MacBook touchpad gestures**: enabled three-finger drag. Three-finger left/right swipes now switch immediately to the next/previous workspace, matching `SUPER + TAB` behavior and avoiding the less-smooth built-in swipe animation.

## Hands-on experience

My Acer Swift 3 had been collecting dust for two years and was nearly unusable: on Windows the fan was constantly roaring. With Omarchy installed, it runs quietly and feels normal again.

The UI/UX is extremely clean and beautiful — minimalism is beautiful in itself.

Everything is text- and code-based, which makes it very agent-friendly. Most configuration tweaks and customizations can be handled with Kimi's help.

On a 2019/20 MacBook, Omarchy feels even more complete: macOS-style natural scrolling and three-finger drag make one-handed touchpad use comfortable, and three-finger horizontal workspace switching is arguably faster than keyboard shortcuts. The Retina display also makes the Linux desktop and terminal text look crisp, and the Touch Bar above the keyboard works normally.

![Omarchy and Kimi running on the Acer Swift 3](/img/omarchy820.jpg)
