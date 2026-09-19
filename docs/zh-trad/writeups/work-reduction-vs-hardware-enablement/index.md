# 工作減少 vs 硬件賦能

> 優化可以分爲兩類：工作減少與硬件賦能。

---

LLMS index: [llms.txt](/llms.txt)

---

> 優化可以分爲兩類：工作減少（work reduction）與硬件賦能（hardware enablement）。

就計算機編程而言，“工作”（work）大體上是對程序需要做多少事情的一個總體度量。

工作優化的思路是減少程序需要做的事情的總量。常用的技術包括：`approximation`、`tail-recursion elimination`、`coarsening/refining recursion`、`inlining`、`loop fusion`、`loop unrolling`、`hoisting`、`short-circuiting`、`common-subexpression elimination`、`compile-time initialization`、`compile-time evaluation`、`exploiting sparsity`、`caching`、`pre-computation` 以及 `bit hacks`。

減少工作無疑是降低整體運行時間的重要啓發式原則，但它並不是運行時間的唯一決定因素；它並沒有涵蓋計算機編程的全貌，因爲它沒有處理計算機硬件本身的複雜特性。

要深入研究能夠釋放硬件潛力的架構改進，我們必須深入到硬件與微架構（micro-architecture）的諸多方面：`the ISA`、`pipeline stages`、`superscalar processing`、`out-of-order execution`、`paging`、`caching`、`vectorization`、`speculation`、`hardware prefetching`、`branch prediction` 等等。

> 縱觀歷史，計算機架構提升性能的手段無外乎利用局部性（locality）或並行性（parallelism）。

爲了利用局部性，內存層級（`registers`->`L1/L2/L3 caches`->`local DRAM`->`remote DRAM`->`PMem`->`SSD`）被做得更深，以掩蓋性能問題；硬件預取器和分支預測器被用來預測即將到來的訪問，把數據或指令移到離處理器更近的地方。作爲程序員，我們工作在計算機架構層之上的一層，我們能做的是編寫 NUMA 感知、緩存對齊、最好還能向量化的代碼，保持規則的數據訪問模式，並輔以恰當的軟件預取。

爲了利用並行性，人們引入了帶 micro-ops 的超標量亂序流水線、向量硬件和多核。相應地，我們需要通過 `bit tricks`、`ILP`（指令級並行，Instruction-level parallelism）、`AVX`/`SSE`、`AMX`、多線程/多進程編程，以及將計算卸載到 `DPU` 或 `GPU` 這類加速器上，讓所有這些硬件都保持忙碌。

讓我們進一步探討 `ILP`，因爲它與處理器內部的 μ-arch 設計——如亂序執行、數據旁路（data bypassing）、寄存器重命名（register renaming）等等——聯繫更爲緊密。要以編程方式利用 CPU 的 μ-arch，我們可以：(1) 使用相互獨立的功能單元；(2) 爲分支預測添加 likely/unlikely 提示；(3) 提前打破數據流圖中的依賴，以減少數據冒險（data hazard）。
