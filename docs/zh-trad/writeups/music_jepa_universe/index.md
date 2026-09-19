# Music JEPA 和三體宇宙

> 生存維數，維度坍塌，死維復活，宇宙規律，田園宇宙，倖存子空間，威懾項λ。

---

LLMS index: [llms.txt](/llms.txt)

---

前文[Music JEPA Regularizers](https://jipeng4974.github.io/writeups/music_jepa_reg/)兼具數學之嚴格、工程之呆板和AI之平庸，寫完總覺得缺了什麼沒表達出來。本文嘗試用三體宇宙進行儘可能有效的類比，嘗試將Music JEPA思想實驗過程中，我的那些穿越星雲的想像，以及從驚詫到歡欣的心路歷程分享出來。

## 多重概念映射表

| Music JEPA SSL | 三體宇宙 |
|---|---|
| embedding space | 宇宙 |
| 音樂表徵模型 | 整體意義上的文明 |
| reward hacking | 在殘酷宇宙中求生並被迫作弊的文明生存策略 |
| effective ranks | 文明實際展開生存的維數 |
| invariance loss（對齊規則） | 宇宙規律 |
| 維度坍縮（erank下降） | 二向箔/文明向低維遷移 |
| 虛假高斯繁榮的倖存子空間/低秩盲區 | 黑域/低維小宇宙 |
| collapsed ranks（$\sigma\to 0$ ）|  死維度/熱寂維度 |  
| $N(0,I_{512})$ isotropic Gaussian理想分佈 | 早期高維宇宙的田園時代 |
| 復活死維度的正則項 | 放棄小宇宙，歸還質量的歸零運動 |
| 對比學習項拉大異類距離 | 個體文明之間的黑暗森林法則 |
| $\lambda_{\text{reg}}$ | 歸零主義的影響力 |
| $\lambda_{\text{infonce}}$  | 執劍人的威懾度 |

## Music JEPA的訓練目標和實驗現象
Music JEPA訓練目標包括：
- 目標1: 讓模型學會anchor music和degraded music之間的invariance，即服從宇宙規律。
- 目標2: 讓模型embedding space的分佈趨近isotropic Gaussian。
- 此外，實踐中還往往可引入目標3：額外設置一個對比學習項，讓模型學會將一些明顯的negatives和anchor拉開距離——雖然這與JEPA無關，但對比學習簡單有力，能在目標1和目標2之外，額外提供一些不會錯得離譜的訓練動力。

前文[Music JEPA Regularizers](https://jipeng4974.github.io/writeups/music_jepa_reg/)中提到了兩個有趣的實驗現象：
- SIGReg、VISReg能防坍塌，但均難以復活死維，VISReg稍強，但常規λ調度下，提升erank方面的訓練動力仍明顯不足。
- 如果給VISReg一個超超高的λ——全力以赴奔向isotropic Gaussian，會發現它不僅不復活死維，反而會導致維度坍塌，VISReg loss下降的同時，erank緩慢陰跌。也就是說訓練陷入了局部低秩最優，或者說“低秩盲區”。

補充一些新觀察到的現象：
- 但常規λ調度下，VISReg和SIGReg的eranks會在特定數值K收斂，徹底失去增長動力，這個K實際上接近白化後表徵流形的內稟維度上限。
    - VISReg和SIGReg都確切具有死維復活能力，雖然速率和穩定性略有不同。
    - 除了invariance對齊規則外，參與訓練的內容豐富度也影響表徵流形的內稟維度上限。
- 超強λ約束和激進invariance目標約束下，模型有可能在訓練的部分時間（前中段）偶然爆發出invariance loss降低和erank提升的理想疊加態，但這種爆發不穩定，訓練步數增加後會屢次發生坍塌。

## 宇宙規律是最殘酷的武器，律法不可過苛
星際文明間的黑暗森林法則有些黑暗但並不殘酷，殘酷的是不可阻擋的宇宙自由度流失（熱寂、降維、光速降低）。

類似地，在Music JEPA自監督表徵學習中，真正殘酷的是目標1 invariance learning。過分激進的invariance loss會導致模型進行痛苦的低維逃逸，通過對錶徵流形進行信息刪除，削足適履地滿足invariance對齊，這意味着更嚴重的內稟維度坍塌，遠比contrastive loss難以避免的false negative樣本對geometry的局部扭曲更嚴重。

所以**不要設定太誇張的invariance目標**，此前我在一次實驗中強制要求15s anchor裁切出來的1.5s～15s子集在強變速變調加噪失真下仍然能和anchor對齊，於是出現了壯觀的低維逃逸，表徵流形內稟維度的損失靠超高$λ_{reg}$也救不回來——eranks上升曲線放緩並不是正則項設計缺陷，後續的白化實驗證明了實則已經觸及這個invariance規則下對應的維度上限了。

秦律，失期法皆斬，過於殘酷的律法不但沒有帶來更堅固的秩序，反而會招至沸反盈天。

因此JEPA語境下Curriculum Learning的真正意義，不在於讓learner從簡單到難，逐步攀登，學會更神奇的對齊任務[^1]。而在於當發現learner在雙重約束下，不得不痛苦自殘、向低維逃逸時，適時地放棄後續的課程，從而規避災難性的坍塌。

## 追逐Isotropic Gaussian和“田園時代”，而避免陷入低秩盲區
早期高維三體宇宙擁有充沛而均勻的自由度，沒有黑域，沒有維度戰爭，沒有過分擁擠後的黑暗森林戰爭，沒有偏安一隅的小宇宙，文明有充裕的生存空間，因此被稱作“田園時代”。三體宇宙中的歸零主義的目的是歸還維度，修復大宇宙，迴歸“田園時代”。

Music JEPA的目標2即迴歸isotropic Gaussian的“田園時代”。爲何isotropic Gaussian是理想分佈？因爲數學上它已被證明是是最適合所有下游任務的理想分佈。Embedding space接近isotropic Gaussian方可自稱是通用foundation model。以檢索任務爲例，哪怕不做任何negative mining，不做任何對比學習，isotropic Gaussian也能天然保證negative cosine p99999足夠低——表徵自然而然就勝任大規模向量庫的檢索。

前文中介紹的3種正則項中，SIGReg和VISReg都直接以isotropic Gaussian爲目標，VICReg有更強的復活死維，撐erank下限的能力，只是不保證形狀。因此將VICReg和VISReg結合起來使用，或在VISReg中引入VICReg的協方差項（完全不影響isotropic Gaussian目標），有助於避免JEPA學習過程中陷入“低秩盲區”[^2]。

所謂“低秩局部最優/低秩盲區”是一種倖存子空間的虛假高斯繁榮的幻像。在激進invariance 目標和強力$λ_{reg}$的雙重約束下，SIGReg和VISReg均有幾率陷入這種局部最優——看起來正則項loss的確在降，但erank增長乏力，甚至緩慢陰跌。看着監控裏embedding space上的維度一個一個死掉，活着的維度卻更擁擠更熱鬧，昂然有序地朝着伊甸園形態逼近——這一幕讓人不經想起三體宇宙中主動將自身降維，向低維度逃逸的文明：低維度小宇宙中秩序在新生，文明熙熙攘攘，低維生物視角下儼然是毫無破綻的伊甸園，但高維大宇宙遺民視角下，一切舊世代文明痕跡都在灰飛煙滅。


## 引入對比學習項——三重張力的微妙平衡
Music JEPA 實踐中的目標3不夠JEPA，但行之有效，負面效果可控，因此是可選項。

類比到三體宇宙，引入對比學習項相當於在已經面臨必然淪亡宿命，渴望迴歸“田園時代”的諸文明之間，引入額外的黑暗森林法則。這第三種張力或許在宇宙消亡的大圖景下無足輕重，但在特定時間節點（無法投入更多GPU hours）對特定文明個體（特定下游任務）來說卻也能發揮決定性作用。

代價是引入新的超參$λ_{infonce}$，需謹慎控制，達到某種微妙平衡。
- $λ_{infonce}$太低或爲0，則在isotropic Gaussian 的田園時代真正降臨之前，永遠會有異常高的negative cos。恰如程心作爲執劍人。
- $λ_{infonce}$足夠高，相當於威懾等級足夠高，從而保住異類距離的下限。恰如羅輯作爲執劍人。
- $λ_{infonce}$太高則喧賓奪主，削弱JEPA的主目標。在微小擾動下選擇同歸於盡，也會斷絕文明健康演進的生路。

三重張力作用下，什麼纔是理想結局？
- 從 Music JEPA 的角度說，理想結局是模型學會了難度適中的 degraded crop invariance 對齊，同時讓 embedding 分佈大致接近 isotropic Gaussian，證得“基礎模型”。
- 從三體宇宙文明的角度說，理想結局是文明適應光速不變，熵增熱寂，黑暗森林等宇宙規則和推論，同時齊心協力執行歸零行動，復活大宇宙死維度，迴歸“田園時代”。

[^1]: 在苛刻的invariance目標下，ViT的確能學到無中生有的超短時長和長程旋律的對齊，甚至同一首歌的無關片段的對齊。
[^2]: 當然可能存在更數學、更嚴謹、更完備的正則方法克服這種低秩盲區，只是我當下缺乏足夠的數學功底和時間去探索。
