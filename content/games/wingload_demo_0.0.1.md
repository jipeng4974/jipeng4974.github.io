---
title: "Wingload 0.0.1: Playable Demo"
date: 2026-08-24
description: "A vertical scrolling shooter where you redesign your ship's cabin layout between waves — playable web demo."
type: docs
comments: false
---

**Wingload** is a roguelike, vertical scrolling shooter (shmup) I'm building in Godot. The twist: your ship is not a fixed sprite but a grid of **cabins**, and between waves you can retrofit it — install or remove propulsion, weapon, sensor, and AI systems, bolt on new cabins salvaged from enemies, or strip cabins off to shrink your hitbox and mass. Enemy fire can penetrate and explode, so a fat ship is a liability: bigger isn't better.

Flight itself is part of the challenge. There are no arrow keys — you control each engine's throttle gear with `1`/`2` (`3`/`4`), and lateral movement only comes from the torque of uneven thrust rotating the hull. Weapons fire automatically.

**Controls:**

| Input | Action |
|---|---|
| `1` `2` (`3` `4`) | Cycle each engine's gear 0→1→2→3→0 (gear 1 = hover against the scroll) |
| Differential thrust | Steering — the gear difference between left/right engines rotates the ship |
| `Tab` | Open the retrofit screen (only while no enemies are on the field) |
| `Esc` | Abandon the run (or close the retrofit screen) |
| `R` | Restart after victory/defeat |

Win by clearing all 6 waves and destroying the boss mothership; lose when all crew cabins (green) are destroyed. The starting hull's 4 crew cabins are effectively your 4 lives.

![Wingload 0.0.1 gameplay](/img/wingload_0.0.1.png)

The demo is a ~40MB WebAssembly build, so it only downloads after you click:

{{< gamedemo src="/games/wingload/0.0.1/" play="Play the demo" >}}
