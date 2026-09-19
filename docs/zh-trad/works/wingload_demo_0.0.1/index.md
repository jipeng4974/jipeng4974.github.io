# Wingload 0.0.1：網頁試玩

> 豎版飛行射擊 + 機艙改裝：在波次之間重新設計你的機體拓撲 —— 可直接在瀏覽器裏試玩。

---

LLMS index: [llms.txt](/llms.txt)

---

**Wingload** 是我用 Godot 開發的一款 Roguelike 豎版飛行射擊遊戲（vertical shmup）。它的核心機制是**機艙改裝（retrofit）**：你的戰機不是一張固定貼圖，而是由若干機艙組成的拓撲結構。在戰鬥間隙，你可以拆裝動力、火力、感知、智能等系統，用敵人掉落的機艙擴充機體，也可以精簡機艙來縮小被彈面、降低質量。敵方火力帶有穿透和爆炸屬性，一發可能掀掉多個機艙——機體不是越大越好。

開飛機本身也是挑戰的一部分：沒有方向鍵，你用 `1`/`2`（`3`/`4`）控制各引擎檔位，左右引擎的差速產生力矩來旋轉機身，橫向移動只能靠轉向獲得。火力系統自動開火。

**操作：**

| 輸入 | 作用 |
|---|---|
| `1` `2`（`3` `4`） | 循環切換對應引擎檔位 0→1→2→3→0（1 檔 = 抵消卷軸懸停） |
| 差速 | 轉向——左右引擎檔位差產生力矩 |
| `Tab` | 打開機艙改裝（僅戰場無敵機時可用） |
| `Esc` | 放棄本局（改裝界面打開時 = 關閉改裝） |
| `R` | 勝利/失敗後重新開始 |

勝利條件：清完 6 波敵人並擊毀 boss 母艦；失敗條件：乘員艙（綠色）全部損毀。開局機型的 4 個乘員艙就是 4 條命。

![Wingload 0.0.1 遊戲畫面](/img/wingload_0.0.1.png)








<button type="button" class="gamedemo-play" data-target="gamedemo-0">開始試玩</button>
<div id="gamedemo-0" class="gamedemo-overlay" hidden>
  <div class="gamedemo-hint">加載中……</div>
  <iframe data-src="https://assets.wujipeng.com/games/wingload/0.0.1/index.html" title="開始試玩" allow="autoplay; fullscreen; gamepad" allowfullscreen></iframe>
  <button type="button" class="gamedemo-close" aria-label="Close">✕</button>
</div>
