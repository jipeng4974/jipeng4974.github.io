---
title: "Wingload 0.0.1 Dev Notes: Headless Testing, Porting to GDScript, and the Web Export"
date: 2026-08-24
description: "Field notes from AI-agent-driven Godot development: an in-game test harness, a full C# → GDScript port, the export pipeline with runtime verification, and a gotcha list."
type: docs
comments: false
---

[Wingload 0.0.1]({{< relref "wingload_demo_0.0.1" >}}) was developed end-to-end by an AI agent: the agent wrote the code, but it cannot play the game with its own hands. That constraint shaped the whole workflow — the game must ship with test hooks that a script can drive, otherwise the agent can say "the code is written" but never "the game works." This post covers the three main threads of the 0.0.1 build: a **headless test harness**, a **full C# → GDScript port**, and the **export pipeline with runtime verification**, closing with a gotcha list from the trenches.

## The headless test harness: the only regression net

The core idea is two env-var-gated test hooks in the main scene script, **kept in the shipped code** (zero cost when gated):

- `WINGLOAD_SMOKE=1` — a frame-scripted auto-test: at given frame numbers it simulates key presses, calls debug interfaces to grant gold / drop cabins, spawns enemies and the boss directly, and walks the full retrofit flow. Every step prints `[SMOKE] PASS: ...` / `[SMOKE] FAIL: ...`, then a summary and `get_tree().quit(0)`. The 0.0.1 smoke has 28 checks.
- `WINGLOAD_SHOT=<dir>` — at a given frame, `get_viewport().get_texture().get_image().save_png(...)`, then quit. Screenshots **must run windowed**: headless mode has no rendering, and would only produce empty images.

Design points that matter:

1. **Determinism first.** Random drops and other nondeterminism get bypassed via test-only paths — the smoke disables ambient spawners and replaces random drops with debug grants, otherwise the test goes flaky.
2. **One PASS/FAIL line per check, with final-value assertions.** Example: gold bookkeeping step by step, `gold 500 -> 505, expect 505`. The agent can read the log and pinpoint exactly where it broke.
3. **Test both win and lose paths.** The victory overlay, the defeat overlay, and rejection paths of the pause rules (pressing Tab mid-combat to open the retrofit screen gets refused) are all independent checks.

On top of the harness, verification climbs a ladder — **look at the output of every rung yourself, never just the exit code**:

```bash
# 1. Compile (C# projects; GDScript projects run the import check below instead)
dotnet build

# 2. Headless editor import — catches script errors / resource import errors
godot --headless --editor --quit --path .

# 3. Smoke test — the main logic verification
WINGLOAD_SMOKE=1 godot --headless --path .

# 4. Screenshots — visual verification, the agent looks at the images itself
WINGLOAD_SHOT=/tmp/shots godot --path .

# 5. Soak run — long unscripted run, catches runtime errors the smoke doesn't cover
timeout 420 godot --headless --path .
```

Two easy misreads: the smoke's exit code only means "it ran to completion" — the passing criterion should be "exit 0 AND grep finds N PASS lines and 0 FAIL lines." And in a soak run nobody is piloting, so the ship getting destroyed around wave 4 is normal, not an error. The `resources still in use` warning on exit is usually an ogg still playing at that instant — benign, and it doesn't affect the exit code.

## The export pipeline

### Export templates and CLI export

Distribution-packaged Godot (e.g. Arch's `godot`/`godot-mono`) ships **without** export templates. Download the tpz, unpack it, and place the files in the directory matching your editor version (mind the `.mono` suffix):

- Standard: `~/.local/share/godot/export_templates/4.7.2.stable/`
- Mono: `~/.local/share/godot/export_templates/4.7.2.stable.mono/`

**The mono tpz has no web templates** (because C# can't export to web at all — more below); web exports need the standard tpz, and the two template sets can coexist in separate version directories. After downloading, verify integrity with `file` + `unzip -t` — when a download source errors out, `curl` may happily save the HTML error page and still return 0, leaving you with a few-hundred-byte "fake success" tpz (passing `-f` prevents this).

The CLI export itself is simple:

```bash
godot --headless --path . --export-release <preset name> <output path>
```

Presets live in `export_presets.cfg` (plain text, hand-editable; missing options get defaults). Create the output **directory** yourself with `mkdir -p` — the exporter won't. For C# projects, make sure `dotnet build` passes first; the exporter packs the build artifacts into the pck. With `binary_format/embed_pck=true`, the Linux preset produces a single-file executable.

### Let the export run its own smoke

A successful export ≠ a working export. Since the smoke hook is env-var-gated and ships with the package, you can verify the exported binary directly:

```bash
WINGLOAD_SMOKE=1 ./build/linux/wingload.x86_64 --headless   # 28/28 PASS
```

That is a much stronger statement than "the export succeeded" — it proves the script assembly, resources, and data JSON were all packaged correctly.

## The full C# → GDScript port

### Motivation and decision

0.0.1 started as a C# project, but Godot 4's C# cannot export to Web — the mono editor refuses outright:

> Exporting to Web is currently not supported in Godot 4 when using C#/.NET.

A prototype was shown by the community (godot#106125), but as of 4.7.2 it hasn't landed, and there is **no official timeline**. If you want web, a full port to GDScript is the only path. The port came with a bonus: dropping the mono dependency shrinks the export, removes the `dotnet build` step, and speeds up editor iteration.

One key decision: **no bilingual coexistence**. A scene file references exactly one script; keeping both languages means double maintenance and behavioral drift. Either don't port, or delete every trace of C#.

### Porting order: secure the regression net first

1. **Port the test harness first** (the smoke/shot hooks in Main — 469 lines, 25 frame branches). It is the only safety net during the port. Port it early and get the skeleton green.
2. Pure-logic classes (data records, catalogs, rule engines) — essentially 1:1 transliteration.
3. The node layer (scene tree / UI / input / audio) — APIs map 1:1, but this is where most signal reconnections happen.
4. Scene file rewrites → cleanup → full verification.

### Language mapping table

| C# | GDScript |
|---|---|
| `event Action X` / `event Action<T>` | `signal x` / `signal x(arg)`; **every subscription site must be reconnected by hand** — a missed one is a silent no-op, the compiler won't save you |
| interface (e.g. `IEnemyHitTarget`) | duck typing: `node.has_method("take_hit")`; preserve the original dispatch **order** |
| static class / static field | `class_name` + `static var`/`static func` (Godot 4 supports static members), or an autoload |
| record / record struct | small class or Dictionary |
| `ref`/`out` parameters | return an Array/Dictionary |
| LINQ (First/Any/Where) | loops / `filter()` / `any()` / `all()` |
| type-pattern switch | if-chain + `is` / `has_method`; audit every order-sensitive branch |
| `[Export]` | `@export` |
| `async Task` + `ToSignal(...)` | `await signal_name` |
| enum | `enum Name { A, B }` (nearly identical syntax) |
| JSON (JsonDocument etc.) | `JSON.parse_string()` + Dictionary — the GDScript version is usually **shorter** |
| multiple classes per file | split files or inner classes; mind which class the tscn references |
| Dictionary keyed by node objects | works, but beware dangling keys after nodes are rebuilt/freed |

### Scenes and project files

- Rewrite every `ext_resource path="res://scripts/**.cs"` in all .tscn files to the .gd path (uids don't matter — the editor import regenerates them).
- Deletion list: all `.cs`, all `.cs.uid`, `.csproj`, `.sln`, `.godot/mono/`.
- `project.godot` usually needs **zero changes**.
- After the port, run `godot --headless --editor --quit --path .` once to rebuild the import cache, and read the output for script errors.

### Verification and measured results

Verification order: headless editor import with no script errors → smoke fully PASS (the same check set as the C# version) → screenshot comparison against the old C# captures (desktop + mobile) → a 60s+ non-smoke headless soak → a final grep confirming no .cs files remain and no .tscn references .cs.

Measured: 4,846 lines of C# (34 files) → 4,435 lines of GDScript (45 files — the multi-class C# files got split). GDScript came out shorter, with the JSON catalog sections shrinking the most. Every class kept its C# name as `class_name`, so cross-references read 1:1 and review stays cheap.

For an agent this was a large but **mostly mechanical** task: no threads, no reflection, no heavy generics. The real risk concentrates in two places — signal reconnection and duck-typing dispatch order. The acceptance criterion is always a fully green smoke, never "all files translated."

The port surfaced two new gotchas of its own:

- **`Array.filter()` returns an untyped Array**, which cannot be assigned back into an `Array[Enemy]`-style typed array — the smoke didn't cover it; the 60s soak exposed it. Fixed with a reverse-order `remove_at` loop. Lesson: typed-array assignment compatibility only blows up at runtime, so the soak is not optional.
- `RandomNumberGenerator(seed=42)` produces a different sequence than C#'s `System.Random(42)` — determinism is preserved, but pixel-for-pixel parity with the old build is not (the starfield moved; harmless).

The Web export passed immediately after the port, producing a 38 MB wasm.

## Runtime verification of the web export

Web export success ≠ runs in a browser. Verifying the wasm's actual runtime behavior takes some work:

1. **The local HTTP server must send COOP/COEP headers** — threaded wasm requires cross-origin isolation:

   ```python
   class H(http.server.SimpleHTTPRequestHandler):
       def end_headers(self):
           self.send_header("Cross-Origin-Opener-Policy", "same-origin")
           self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
           super().end_headers()
   ```

2. **Headless Chromium screenshots can't use `--virtual-time-budget`** — it has no effect on wasm startup; the screenshot stays stuck on the loading page forever. The working approach is real waiting over CDP: launch with `--remote-debugging-port`, `Page.navigate`, actually sleep 40s+, then `Page.captureScreenshot`. Also subscribe to `Runtime.consoleAPICalled` to collect the game's print logs — seeing `[WAVES] wave N starting` in the console directly proves the game is running. Node ≥ 21 ships a built-in WebSocket client, so a few dozen lines of script can drive CDP. On software rendering, add `--enable-unsafe-swiftshader --disable-gpu`.

3. **Static hosting needs the nothreads variant.** Threaded wasm requires the server to send COOP/COEP headers, which static hosts like GitHub Pages can't do. Duplicate the Web preset in `export_presets.cfg` and set `variant/thread_support=false` as a separate preset; the output starts fine under a plain `python3 -m http.server` with no special headers (the console shows a "single-threaded" build). The cost is a potential performance hit, so keep it as an optional preset rather than replacing the default.

## Gotcha list

Finally, a list of real traps — each one actually cost time:

**Physics and the scene tree**

- **"Can't change state while flushing queries"**: calling `add_child()` directly inside a physics callback like Area2D's `area_entered` (e.g. dropping gold when an enemy dies) errors out. Fix: `call_deferred("add_child", node)`, or `set_deferred()` for collision properties.
- **pause freezes Timers/Tweens**: the victory overlay pauses the whole tree the instant the boss dies, which froze the originally designed chain of delayed explosions. Fix: play critical feedback fully before pausing, or set `process_mode = ALWAYS` on the overlay node.
- **Instantiation position timing**: setting `position` before `add_child` can get overwritten by `_Ready`/layout — especially visible after the mobile layout translates the scene. Fix: `add_child` first, then set `global_position`.

**Coordinate systems**

- **Out-of-bounds checks belong in local coordinates**: mobile mode translates the whole battlefield container by −360px, and doing despawn checks with `global_position` against a canvas-space rectangle made enemies get destroyed the moment they spawned. Switching uniformly to container-local `position` left desktop (identity transform) behavior unchanged. Lesson: **before comparing coordinates, confirm both sides live in the same coordinate space**.
- **Centroid re-anchoring moves the ship's origin**: when cabins are destroyed, the hull rebuilds around the new centroid and the origin shifts — every hardcoded "spawn a test enemy in front of the ship" coordinate broke with it. Test/spawn logic should read the ship node's live position; never hardcode it.

**Rendering and UI**

- **CanvasLayer (HUD) draws above canvas items**: the 40px HUD bar at the top covered the warning arrows at the top of the game world. Reserve the HUD height when laying out the battlefield.
- **Kenney fonts have no CJK glyphs** — Chinese renders as tofu boxes. The deeper trap: **SystemFont fallback does nothing in web exports** (browsers have no system fonts), so Chinese text that looked fine on desktop turned entirely to tofu on the web. The correct fix is shipping a subsetted font: use fonttools' `pyftsubset` on Noto Sans CJK SC (OFL-licensed) to produce a 103KB otf containing only the 247 CJK characters the game uses, attached uniformly as the Kenney font's fallback. Regenerate the subset whenever new Chinese characters are added. Block characters like ▮/▯ are also missing from Kenney — substitute `#`/`-`.

**Windowing and platform**

- **Tiling window managers ignore `--resolution`** (e.g. Hyprland), forcing the window into odd tiled sizes. Countermeasure: set stretch to `canvas_items` + `expand` so the layout adapts to any size; when you need an exact screenshot size, float and resize the window via the WM's IPC (e.g. hyprctl) before capturing.
- **The headless editor doesn't generate C# project files**: `godot-mono --headless --editor --quit` won't create `.csproj`/`.sln` (that only happens when you add the first C# script in the GUI). Hand-write a few-line csproj instead, with the `Godot.NET.Sdk` version aligned to the engine version.

**Audio**

- **Gunfire needs throttling**: with multiple weapons auto-firing, calling `play()` on every shot smears into noise. Rate-limit per sound type (≤4 plays/second) — no object pool needed.

## Closing

The 0.0.1 workflow distills to one sentence: **keep the test hooks in the shipped code, and make the verification ladder muscle memory**. The harness is the only regression net — every refactor and every port starts by keeping it green, and acceptance always means a fully green smoke plus screenshots you actually looked at, never "the code is written." As for the C# → GDScript port, the biggest takeaway: a large mechanical task suits an agent perfectly — as long as the regression net is solid, the only places needing human review are signal reconnection and dispatch order.
