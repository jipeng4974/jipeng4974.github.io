# Dash: 可擴展哈希

> Dash 論文的主要關注點是曾經風靡一時的 `persistent memory`，但實際上，任何受 `memory bandwidth` 限制的場景都能從中受益。隨着 Intel 砍掉其 `pmem` 業務，`Dash` 方案的意義已經轉移到了普通的 `DRAM` 應用上。

---

LLMS index: [llms.txt](/llms.txt)

---

Dash 論文的主要關注點是曾經風靡一時的 `persistent memory`，但實際上，任何受 `memory bandwidth` 限制的場景都能從中受益。隨着 Intel 砍掉其 `pmem` 業務，`Dash` 方案的意義已經轉移到了普通的 `DRAM` 應用上。

# 動態哈希
論文提出的可擴展哈希表 `Dashtable`，由 `extendible hashing` 演化而來。

`Extendible hashing` 是一種哈希體系，它使用哈希值的前 $N$ 位，在 trie 結構的 `directory` 中查找 bucket。

`global depth` 爲 $N$ 的 `directory` 可以容納 $2^N$ 個 bucket。這意味着 $N$ 是映射該 `directory` 的鍵長。

每個 bucket 還有一個 `local depth` $M(M \le N)$，即此前映射該 `directory` 的鍵長。`local depth` 爲 $M = N$ 的 bucket 恰好被一個 `directory` 條目指向；`local depth` 爲 $M \lt N$ 的 bucket 則被多個 `directory` 條目指向。

要保證每個 item 都有唯一的 bucket 索引，2 個 item 所需的最小 $N$ 爲 1，4 個 item 所需的最小 $N = 2$。每當有新 item 加入 bucket，如果 bucket 中的 item 數量超過某個閾值，就會觸發一次 rehashing：將該 bucket 一分爲二。因此，這種方案中的 rehashing 不需要 stop the world、做全表掃描和拷貝，而是增量完成的。

與 `extendible hashing` 類似，`linear hashing` 同樣使用 `directory` 來組織和尋址 bucket。區別在於 split 的控制方式。在 `linear hashing` 中，通常只有當 load factor 超過閾值時纔會發生 split，且待 split 的 bucket 是以“線性”的方式選出的。

# 面向 Extendible Hashing 的 Dash
## 概覽
![dash_eh](https://wujipeng.com/img/dash_eh.png)

在 `Dash-EH` 中，每個 `directory` 條目指向一個 `segment`，它由固定數量的普通 bucket 和 stash bucket 組成。一個 `segment` 可以看作一個大小恆定的子哈希表。所謂 stash bucket 與普通 bucket 佈局相同，負責存儲溢出記錄。

![dash_eh](https://wujipeng.com/img/dash_eh_bucket.png)

`Dash-EH` 的核心思想是在元數據上多花一點空間，換來基於 fingerprint 的更快 probing，以及基於 version lock 的輕量級併發控制。

如上圖所示，在一個 `Dash-EH` bucket 內部，前 32 字節是元數據，包括 version lock、計數器、alloc bitmap、用於負載均衡的 membership bitmap，以及 18 個用於 bucket probing 的單字節 fingerprint（其中 14 個對應 bucket 內的 slot，4 個對應原本哈希到此 bucket 的溢出記錄）。緊隨其後的是 $16(Bytes) \times 14 (records) = 224 Bytes$ 的 payload，存放 14 條 16 字節的記錄。

## Fingerprinting
Bucket probing 指在 bucket 中查找某個 slot，是哈希表的基本操作，`search`、`insert` 和 `delete` 都需要它來定位特定的 key。傳統的 probing 需要線性掃描，這在 `PMem` 上天然就慢，而且當被查找的 key 不存在時，掃描可能完全是多餘的。`Dash-EH` 採用 fingerprinting 來減少不必要的掃描。fingerprint 是 key 的哈希值的最低一個字節。probing 線程在查找某個 key 時，會先檢查 bucket 元數據中是否有與該 key 匹配的 fingerprint，從而跳過沒有任何 fingerprint 匹配的 bucket。

Fingerprinting 主要惠及 negative（key 不存在）的 `search`。`Dash` 論文還聲稱，fingerprinting 使得使用跨越 2 條以上 cacheline 的更大 bucket 成爲可能。但對此我得持保留態度。論文自己用的是 256B 的設置，DragonFly 的實現[^1] 也是如此。理論上，更大的 bucket 確實能容忍更多衝突、提高 load factor；然而，這可能以在一定程度上犧牲 locality 爲代價——在哈希表中，你不會希望訪問一次 bucket 要加載多次。

## Bucket 負載均衡
Segmentation 通過減小 `directory` 的體積來降低其 cache miss。在 `extendible hashing` 方案中，如果 segment 裏任意一個 bucket 滿了，整個 segment 都需要 split，即使其他 bucket 可能還有很多空閒空間。

爲了避免頻繁的 segment split，`Dash-EH` 的算法設計引入了 bucket 負載均衡。對於 `insert` 操作，`Dash-EH` 會同時探測 bucket $B_b$ 和 $B_{b+1}$，然後插入較不滿的那個。如果 $B_b$ 和 $B_{b+1}$ 都滿了，`Dash-EH` 會嘗試把 $B_{b+1}$ 中的一條“native record”挪到 $B_{b+2}$，或者把 $B_b$ 中的一條“rebalanced record”挪回它原本所屬的 $B_{b-1}$。

每個 bucket 的 membership bitmap 用於判斷一條記錄是 rebalanced 還是 native。如果 membership bitmap 中某一位被置位，那麼對應的 key 並非直接哈希進這個 bucket（native），而是由於重均衡被放到這裏的（rebalanced）。

如果 `insert` 和 displacement 都失敗了，`Dash-EH` 就使出最後的手段——stashing。每個 segment 有固定數量的 stash bucket 來容納這些溢出記錄。探測 stash bucket 會給 negative `search` 和 `insert`（需要做唯一性檢查）帶來顯著開銷。爲了解決這個問題，每個普通 bucket 預留了若干元數據字段：爲存放在 stash bucket 中的溢出記錄保留 4 個溢出 fingerprint；一個溢出 bit 表示是否存在溢出。這樣，如果一個 bucket 沒有溢出，`search`/`insert` 操作就不必探測 stash bucket。不過，保持少量的 stash bucket 仍然是明智的。論文聲稱“每個 segment 使用 2–4 個 stash bucket，可以將 load factor 提升到 90% 以上，而不會帶來顯著開銷”。在 Dragonfly 的 Dashtable 中，每個 segment 有 56 個普通 bucket 和 4 個 stash bucket。

## 輕量級併發控制
`Dash-EH` 的輕量級併發控制在當今的 `many-core` 架構上天然具有良好的擴展性，性能優於傳統的 bucket 級共享鎖。

寫操作沿用傳統的 bucket 級加鎖方式，通過對一個 lock bit 執行 CAS 來鎖住受影響的 bucket。寫完成後，寫線程復位 lock bit，並將該 bucket 的版本號加一。

另一方面，讀操作被設計爲 lock-free 的。讀之前，讀線程先獲取 lock word 的快照，等待鎖被釋放，然後在不持有任何鎖的情況下繼續讀取。讀完後，它會再次檢查 lock word，確認版本號保持不變。如果版本號變了，就重試整個操作。

# 面向 Linear Hashing 的 Dash
Dash 論文還提出了 `Dash-LH`，一種支持 Dash 的 linear hashing 方案，構建於 `Dash-EH` 所用的構建塊之上，比如均衡的 `insert`/`displacement`、fingerprinting 和樂觀併發——畢竟它們大體上是正交的。主要區別在於，`Dash-LH` 以線性的方式 split 指針所指向的 segment。

傳統的 `linear hashing` 用鏈表（linklist）把溢出記錄串起來。在 `Dash-LH` 中，改用 stash bucket，對 cache 更友好。不過它仍需要把這些 stash bucket 串成鏈，但這仍然比把單條記錄串成鏈好得多。



[^1]: [Dragonfly 中的 Dashtable](https://github.com/dragonflydb/dragonfly/blob/main/docs/dashtable.md)
