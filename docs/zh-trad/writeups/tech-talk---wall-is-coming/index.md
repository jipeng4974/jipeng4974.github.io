# Tech Talk: Wall is Coming

> Tech Talk文稿，梳理內存牆問題的歷史淵源，嘗試給出對優化空間的理解，推導出相應的啓發性策略，並列舉一些訪存優化技術。

---

LLMS index: [llms.txt](/llms.txt)

---

# 內存牆和登納德定律失效
“內存牆（the Memory Wall）”和“登納德定律失效（the break down of Dennard Scaling）”是計算生態演化中的兩個核心矛盾。
- 爲了對抗Dennard Scaling的失效，計算硬件的架構從單核向多核、衆核突圍。
- 爲了掩蓋內存牆問題，內存層級（memory hierarchy）變得越來越深，off-chip互連帶寬不得不迅速提高。

起初單核到多核是鉅變，逼迫軟件架構進行痛苦重構，併發編程問題木秀於林，因此被近20年的工業界實踐+學術界研究集火秒了——如今我們有多線程編程範式、異步回調範式、goroutine式的有棧協程、C++/Rust的async/await無棧協程、lockfree/waitfree數據結構等諸多工具。
相較而言，這幾十年來內存牆問題由於其隱蔽性、不緊急性、棘手性，不僅沒得到妥善解決，反而根深蒂固，愈發遮掩不住，暴露在軟件工程師面前，因此這次分享的重點是內存牆。

不過在進入正題之前，還需先介紹一下數據中心硬件和微處理器架構演化的些許背景。

# 21世紀數據中心硬件的演化簡史
## 00s，Commodity Computing時代
遠古時期我們並不會說“數據中心硬件”，因爲還不存在現代意義上的互聯網產業，也就沒有現代意義上的數據中心。自然也就沒有“for 大數據/雲/Edge/AI”的營銷話術，更多的是強調用廉價、不可靠、大規模的commodity hardware搭建分佈式有狀態系統，Google在這方面做了開創性的嘗試。

> 相對IBM mainframe、super computer而言，當年x86的廉價是一目瞭然的。Google草創之初用的是奔騰2。
> 銀行業、DAPRA成就了IBM power系列，如今Power9/10機器雖然被Xeon/EPYC完爆，仍然可以靠銀行軟件的祖宗之法不可變和政府訂單苟延殘喘，保有一份niche市場。

![commodity_computing](https://wujipeng.com/img/commodity_computing.png)

10s之前是x86微處理器完全不具備多核可擴展性的年代。
- `FSB`(front-side bus)是罪魁禍首。下圖架構中，內存總線和PCIe總線需共享一個`FSB`才能與CPU相連，導致`FSB`成爲瓶頸，CPU數量不具備橫向擴展性。
- 當年的`PCIe`還是1.0（03年初代`PCIe`），帶寬和lane數都相當有限，即使強上多核，網絡IO、磁盤IO也跟不上。
![fsb](https://wujipeng.com/img/fsb.png)

零零年代恰逢摩爾定律逐漸在單核語境失效，專注提升芯片性能在2004年後不再可行，硬件廠商不得不在架構上向多核方向突圍。
![clock](https://wujipeng.com/img/clock.png)

## 一零年代初，多核時代
2012年的Intel Xeon E5-2600 V1 32nm Sandy Bridge是里程碑式的服務器產品，移除`FSB`（這個實際上在09年的Nehalem機器上就已經做了）、引入`QPI`/`DMI`取代`FSB`、PCIe2.0，使微架構獲取多核可擴展性。
> E5 family算是耳熟能詳，雖然普遍要到壽命極限，但至今應該仍有很多公司在用。
> 當年買電腦時，所謂二代i3/i5/i7就是`SandyBridge`，和一代i5/i7的`Nehalem`有代差。

`Sandy Bridge`之後是`Haswell`，變化不大。`Haswell`之後是`Broadwell`，環狀拓撲`Broadwell Ring`即得名於此。
![clock](https://wujipeng.com/img/broadwell_ring.png)
再後面就是Skylake，開始冠上Scalable之名了，從多核走向衆核，從近代走到現代。

## 一零年代末~二零年代初，衆核時代
17年Intel推出了1st gen Xeon Scalable，`Skylake`，採用了Mesh Architecture。見Things are getting meshy
同時期AMD也推出了ENYC 7001，算是打破了Xeon的壟斷局面。在US-TTP機房我們就有不少AMD機器。

`Skylake`的升級版`Ice Lake`並未順利孵化，因爲18年出了Meltdown/Spectre的大新聞（speculative execution的安全漏洞），於是在`Skylake`上修了漏洞，19年推出`Cascade Lake`作爲2nd-Gen Xeon。原本的順位繼承者`Ice Lake`在21年姍姍來遲，變成了第三代。

`Cooper Lake`和`Ice Lake`同代，都被稱爲第三代，但實際上架構和`Ice Lake`不同，是基於`Skylake`改的，專爲多socket(4~8s)設計，相比general-purpose的`Ice Lake`， `Cooper Lake`稍稍超出commodity hardware範疇，估計是想賣給特定的專用計算領域，用來替代老舊的UNIX系統，比如`Oracle Solaris`，`IBM AIX`。互聯網場景下我們還是傾向於橫向擴容而不是縱向擴容。

目前數據中心應用的主力機型是`Ice Lake`、`Cascade Lake`機器，23~24年起計算/訪存密集的場景則會逐步用到第四代Xeon：`Sapphire Rapids`機器。
19年20年我們還零零散散有一些1st gen Xeon Scalable Gold機器，後來很快就汰換掉了。
![clock](https://wujipeng.com/img/broadwell_ring.png)

`Ice Lake`和`Cascade Lake`都是monolithic mesh設計。
- 這裏monolithic是相對於chiplet/tile-based而言的單個huge die承載many-core的範式；
- 這裏mesh是相對於此前E5時代Broadwell Ring環狀拓撲而言的網狀拓撲。

二者差異主要在製程（10nm vs 14nm）、最大核心數、PCIe路數、內存通道數（單socket支持的DIMM[^1]數 16 vs 12）。

24年起開始交付的4th-Gen `Sapphire Rapids`的芯片架構從mono-die轉型爲更類似AMD的multi-die(Intel自稱是tile-based），微架構從sunny cove更新到golden cove（tpause指令可用於優化spinlock），配置相當華麗，支持先進互連協議（`PCIe5`/`CXL`），支持DDR5，新指令集`AMX`，頂配還有3D堆疊的on-package HBM[^2]。

## 一些總結
### 硬件生態裏，生命和環境也是相互塑造的
PC用戶成就了繁榮、易獲得、標準化的x86 商用硬件市場。商用硬件集羣剛好又適應互聯網workloads（沒有大量浮點數計算，integer server爲主，但數據量極爲龐大），纔有了分佈式系統支撐的現代數據中心，再然後纔有硬件廠商爲數據中心定製優化的服務器硬件，比如Scalable Xeon。

PC玩家成就了Nvidia GPU，GPU恰好適應AI workloads，於是有了各種MLSys和GPGPU應用。然後纔有Nvidia在加速計算方向上的投入。有了A100之後，ChatGPT訓練就是在A100+IB network基礎上量體裁衣做的大規模模型並行。隨後又因爲ChatGPT的轟動效應，反哺了高端GPU產業。

生態位很難被人爲設計、憑空創造，比如Intel Optane PMem，佔據了非常合理的生態位，很多系統方向的研究都是靠PMem發的論文，因爲它太合理了，彌補了磁盤和內存之間的空缺。但還是因爲需求側跟不上，在22年被砍掉了。究其根本，PMem在AI訓推場景下沒啥用，在主流的搜廣推應用上不能帶來顯著成本優勢，在交易場景和分析型場景要麼比不上磁盤+內存，要麼不值得投入人力去大改架構。
### 現代硬件特徵
- 核心數衆多
- 設法緩解Memory Wall問題——近20年來DRAM cycle time每年縮減的速率與摩爾定律對比呈相對停滯，cache miss的後果愈發嚴重。
  - Memory hierarchy變深：cache變多，最新的頂配SPR機器還增加了on-package HBM，本地內存之外還有遠端NUMA node，內存下面還可以有PMem、SSD，本地節點之外還有局域網/雲端節點。總之，是通過增加層級隱藏Memory Wall問題。
  - Off-package/chip-to-chip互連帶寬提升：跟上核心數增多帶來的IO需求，緩解批量讀寫場景下的Memory Wall問題。
### 黑盒化和白盒化同時發生
現代硬件對工程能力提出了更苛刻的要求——通過off-CPU analysis、data-oriented cache-friendly設計、手動內存管理、甚至手動prefetching才能真正釋放其性能潛力。

但這和現代軟件工程愈發簡化的發展方向背道而馳——通過runtime、虛擬機、動態語言、微服務範式的職責劃分、基於hypervisor的虛擬化和容器化等手段解放程序員心智。

這種背道而馳使現代軟件實踐走向兩條岔路，一個是以白盒化爲手段的基礎設施建設：更好的hypervisor、更好的mlsys、高性能檢索、高性能存儲、高性能網絡，另一個是以黑盒化爲目標的上層應用：利用虛擬化、容器化、微服務化、動態語言、runtime語言的便捷性提升productivity。

# 現代硬件上的性能工程實踐
## 對優化空間的理解
性能工程即有系統方法論指導的軟件優化實踐。
可以從兩個視角分解“優化任務”：
- 優化 = 減少算法總工作量 + 硬件使能
- 優化 = 減少運行時間 = 減少CPU時間 + 減少阻塞時間
“算法改良”和“硬件使能”接近正交，“on-CPU time”和“off-CPU time”又大體互補，因此可以爲優化空間$W$構造正交基{硬件使能$x$，算法優化$y$，算術密度$z$}，則$W = \{[x,y,z] \in R^3 | 0 \le z \le 1 \}$。

<div id="damn"><svg width="480" height="420"></svg></div>

## 推導出的一些Heuristics
基於對優化空間完整圖景的理解，能推導出一些性能工程的Heuristics：

- 應首先確定程序的算術密度
  - 爲何區分on-CPU vs off-CPU至關重要？因爲你的100%忙碌的CPU可能並不忙碌。CPU profiling僅描述完整圖景的一部分，甚至一小部分。常用的CPU利用率這一指標具有欺騙性和迷惑性，實則既包括on-CPU計算也包括off-CPU阻塞。如果存在嚴重訪存瓶頸，100%的CPU的superscalar pipeline裏全是stall的空泡，CPU的各類算術邏輯單元、SIMD/AMX等專用計算硬件都在空等。
  - 當然off-CPU過高也可能是磁盤/網絡IO密集導致的，但這類IO-bound應用往往不是成本大頭，還不足以興師動衆地做優化，除非是專門做存儲或專門搞網絡的infra部門才需要關心。此外還有可能是代碼寫得有問題，鎖粒度太大，過度持有鎖，同樣造成off-CPU 比例過高。
- 如今的內存應被視爲外設
  - 曾經爲慢速外設準備的數據結構，如今適合內存場景：
    - C++的有序map是用紅黑樹實現的，Rust選擇了B+樹，以便利用其更好的locality，因爲如今的內存已經和幾十年前的磁盤一樣慢得不可容忍了。
    - DashTable這種原本用在PMem（非易適性內存，內存帶寬遠低於DRAM）上的數據結構，如今被DragonFly拿來用在內存數據庫上，性能遠超Redis/Memcached。
  - 因此需將內存層級白盒化，充分發揮硬件潛力。
- 避免和編譯器優化撞車
  - 不需要用移位操作或其他彙編指令優化乘除法。乘以8必然會被自動優化成左移動3。除法也會被優化成乘加。下面的例子是非常古老的編譯器對 volatile int y = x / 71的處理。編譯器優化乘法還會用LEA指令，LEA原本旨在加速小結構體數組的成員地址計算，但實際上也能用來加速乘法，比如乘以5可以寫成lea eax, [eax*4 + eax]，利用現成的電路就比較快，算是硬件使能領域裏編譯器幫你做好的工作。
    ```assembly
    // volatile int y = x / 71;8b 0c 24        mov ecx, DWORD PTR _x$[esp+8] ; load x into ecx
    mov eax, -423447479 ; magic happens starting here...
    imul ecx            ; edx:eax = x * 0xe6c2b44903 d1           add edx, ecx        ; edx = x + edx
    sar edx, 6          ; edx >>= 6 (with sign fill)

    mov eax, edx        ; eax = edx
    shr eax, 31         ; eax >>= 31 (no sign fill)
    add eax, edx        ; eax += edx

    mov DWORD PTR _y$[esp+8], eax
    ```
  
  - 可以放心去做手動SIMD。手動SIMD和LLVM自動向量化優化位於不同抽象層次，基本上也不必對LLVM的自動向量化抱有期望。短期內見不到程序語言和庫層面對SIMD進行良好抽象的希望。
    - 手動SIMD需要程序員根據目標機器微架構型號選擇合適的指令（不同微架構的各種simd指令的latency各不相同）；
    - 選擇恰當的simd size(並非越大越好，而且不同size對應不同的shuffle)；
    - 還要處理最後不滿一個batch的數據導致的各種各樣的corner cases；
    - 對數據地址進行對齊或對不對齊數據進行容忍。
  - 用最新的編譯器，用O3，很多細節就不必手動優化，如Copy Ellison，Tail-recursion Elimination，甚至Mutually recursion Elimination，Inlining，多數Loop優化——Loop unrolling(+步長)/fission(逆fusion)/tiling(cache blocking)/unswitching(內層去分支化)/自動向量化/interchange。
  - 涉及到邏輯和具體應用場景，沒有標準優化策略的優化仍需自己做，比如Loop fusion，調整遞歸粒度，調整編碼策略（緊湊程度和編解碼開銷的tradeoff）。


## 訪存優化的一些Tricks
- 如何度量算術密度或訪存密度?
  - perf 或 perf_event_open: cache_miss/instructions, 或ipc
  - Intel PCM
  - eBPF tools，比如https://github.com/iovisor/bcc
  - 靜態分析：load/store指令佔比可粗略翻譯算術/訪存密度，但受緩存命中率影響較大。
- 如何根據內存配置設置合適的object padding？
  - 內存配置中有兩個影響應用層的重要概念：內存通道（memory channels）、內存列（memory ranks）。x86架構下內存通道和內存列在內存地址上interleaving，即均勻分佈且遞增。因此RAM可視爲$n_chan \times n_rank$個block組成。其DIMM架構如下圖。
[圖片]
  - 具體的object padding方式可以參考下面這段僞碼，其中64B是cache line size，也恰好是一個block的size。大體思路是先保證內存池裏的地址都是64B的整數倍，再保證下一個對象的block id和n_chan*n_rank互質。
  - 這個padding不讓對象的起始地址反覆命中同一個channel或同一個rank，令下一個對象起始地址落入不同的channel/rank，充分利用不同內存通道、不同內存列，避免通道、列之間的負載不均，提升訪存帶寬。
    ```C  
    static unsigned int object_align(unsigned int obj_size)
    {
            unsigned nchan = get_nchannel();
            unsigned nrank = get_nrank();
            unsigned new_obj_size = (obj_size + 63) / 64; 
            while (get_gcd(new_obj_size, nrank * nchan) != 1)
                    new_obj_size++;
            return new_obj_size * 64; 
    }
    ```
- Cache優化：Cache line對齊避免false sharing；利用cache blocking。
- 規避鎖瓶頸
  - 基於靜態鎖分析或ebpf off-CPU分析，找到過分粗粒度的鎖和過度超期持有的鎖。
  - 尋找更好的併發數據結構：無鎖實現良莠不齊；Many-core scalability最好的永遠是array。
  - Kernel bypassing：規避某些帶鎖的內核實現，如用戶態網絡協議棧替代內核棧。
  - Share-nothing：最極端的做法可以效仿seastar，每個核心只在自己的專用內存上執行單線程代碼，儘量避免CPU-to-CPU traffic，將鎖徹底從代碼裏消除。
- In-register storage：儘量保證簡單函數用到的參數、存放中間產物的容器足夠小，可完全用寄存器存下，編譯器自動就會把所有load/store優化掉。
- 考慮使用預分配內存和棧上靜態結構：不得不訪存時，生命週期較長的對象可考慮使用預分配內存，臨時容器可以考慮根據線上數據分佈設計成按某種規則切分的靜態結構，並放在棧上（棧內存分配只是棧指針移動，而malloc複雜的多，如果說默認版本的malloc還有全局鎖，不夠scalable）。
- 考慮使用編譯期evaluation和全局靜態內存區。
- 利用1GB huge pages存數據，相比4KB默認頁，hugepage所需的page table entry總數大大減少，可顯著減小page table size和tlb size、降低tlb miss和page table walk開銷，提升內存分配的連續性和內存訪問的局部性，這些都有助於提升內存帶寬。
- 利用4MB large pages存代碼，.text segment也可以用更大的頁（不過最大隻支持4MB），ITLB和DTLB一樣，miss也會造成stall。ITLB問題的診斷可藉助https://github.com/intel/iodlr，解決方法把.text移動到已有的large pages裏[^3]或靜態鏈接並使用libhugetlbfs庫。目前尚未看到該優化在真實項目中落地，但看起來相當promising，可參考[Runtime Performance Optimization Blueprint: Large Code Pages](https://www.intel.com/content/dam/develop/external/us/en/documents/runtimeperformanceoptimizationblueprint-largecodepages-q1update.pdf)。
- 尊重NUMA拓撲，避免遠端內存訪問，即UPI traffic。
![numa](https://wujipeng.com/img/NUMA.png)
- 利用新架構特性和新的指令集擴展，如基於AMX實現GEMM的精度和性能優化廣泛用在各種訓推框架，AVX512_IFMA指令擴展做大數乘法已被用在新版本的OpenSSL中，以及基於QAT（QuickAssist）對AES、RSA、ECC等密碼學應用做硬件加速。
- 利用prefetching讓cache變得更聰明。
  - 所謂hardware prefetching就是從內存預取數據到cache（通常是LLC）。Hardware prefetcher有簡單的stride pattern識別邏輯，比如a,a+2,a+4,a+6這種loop是可以被識別的。沒必要刻意觸發hardware prefetching，正常的代碼都可以觸發。但需避免誤觸發hardware prefetching——比如一個橫跨多個cache line的大結構體其實只需要訪問它的前幾個field，但hardware prefetcher誤以爲還要接着讀，就會造成cache pollution。稍微調整下這幾個fields的訪問順序，破壞掉constant stride pattern即可。
  - 找到合適的timing使用software prefetching，預取數據會帶來巨大吞吐性能提升，也能用來隱藏latency，但究竟何時預取，預取哪些數據只能靠反覆嘗試。而一旦timing選錯了反而造成cache污染，導致性能下降。
    - 儘量讓prefetch指令分散（最好和load也分散），夾雜在計算指令中間。如果連續prefetch，那和一堆load一樣，也會造成空泡。
    - 選擇合適的PSD(prefetch scheduling distance)，即提前幾個iteration預取。對計算量大的loop，可以提前1個iteration，對於計算量小的，可能要提前多個iteration。下面這個例子PSD=3。
    ```assembly    
    top_loop:
    prefetchnta [edx + esi + 128*3]
    prefetchnta [edx*4 + esi + 128*3]
    movaps xmm1, [edx + esi] 
    movaps xmm2, [edx*4 + esi]
    movaps xmm3, [edx + esi + 16] 
    movaps xmm4, [edx*4 + esi + 16]
    add esi, 128 
    cmp esi, ecx 
    jl top_loop
    ```

    - 雙層循環時，需注意彌補內外循環切換時的空泡，爲外層循環也做prefetch。

    ```C    
    for (i = 0; i < 100; i++) {  
    for (j = 0; j < 32; j+=8) { 
        prefetch a[i][j+8]  // 最後一次iteration，不需要prefetch
        computation a[i][j] // 第一次a[i+1][j]未預取，會miss
    } 
    }
    // 優化後
    for (i = 0; i < 100; i++) {  
    for (j = 0; j < 24; j+=8) {
        prefetch a[i][j+8]  
        computation a[i][j]  
    }  
    prefetch a[i+1][0]  // 提前準備好 a[i][0]，否則會在a[i][j+8]阻塞
    computation a[i][j] // 最後一個iteration單獨處理，因爲它不需要prefetch。
    }
    ```
- 利用先進互連技術，如`CXL`。

![cxl](https://wujipeng.com/img/spr-cxl.png)


[^1]: DIMM(dual in-line memory module)，即ram stick，內存條，DDR(Double Data Rate)技術的物理具現。
[^2]: https://www.intel.com/content/www/us/en/products/sku/232592/intel-xeon-cpu-max-9480-processor-112-5m-cache-1-90-ghz/specifications.html
[^3]: https://github.com/intel/iodlr/blob/master/large_page-c/large_page.c

<style type="text/css">
svg {
    box-shadow: 0 0 10px #999;
    border-radius: 5px;
}
</style>
<script type="module">
import {
  drag,
  color,
  select,
  range,
  randomUniform,
  randomNormal,
  scaleOrdinal,
  selectAll,
  schemePastel1,
} from "https://cdn.skypack.dev/d3@7.8.5";
import {
    gridPlanes3D,
    points3D,
    lineStrips3D,
} from "https://cdn.skypack.dev/d3-3d@1.0.0";
document.addEventListener("DOMContentLoaded", () => {
    console.log("dom loaded, starts to draw svg ...");
    const width = 480;
    const height = 420;
    const origin = { x: width/2, y: height/2 };
    const offset = origin.x - origin.y;
    const j = 10;
    const scale = 20;
    const key = (d) => d.id;
    const startAngle = Math.PI/2;
    // const startAngle = 0;
    const colorScale = scaleOrdinal(schemePastel1);
    let scatter = [];
    let yLine = [];
    let xLine = [];
    let zLine = [];
    let xGrid = [];
    let beta = 0;
    let alpha = 0;
    let mx, my, mouseX = 0, mouseY = 0;
    const svg = select("svg")
        .call(
          drag()
            .on("drag", dragged)
            .on("start", dragStart)
            .on("end", dragEnd)
        )
        .append("g");
    const grid3d = gridPlanes3D()
        .rows(20)
        .origin(origin)
        .rotateY(startAngle)
        .rotateX(-startAngle)
        .scale(scale);
  const points3d = points3D()
    .origin(origin)
    .rotateY(startAngle)
    .rotateX(-startAngle)
    .scale(scale);
  const yScale3d = lineStrips3D()
      .origin(origin)
      .rotateY(startAngle)
      .rotateX(-startAngle)
      .scale(scale);
  const xScale3d = lineStrips3D()
      .origin(origin)
      .rotateY(startAngle)
      .rotateX(-startAngle)
      .scale(scale);
  const zScale3d = lineStrips3D()
      .origin(origin)
      .rotateY(startAngle)
      .rotateX(-startAngle)
      .scale(scale);
  function processData(data, tt, recolor) {
    /* ----------- GRID ----------- */
    const xGrid = svg.selectAll("path.grid").data(data[0], key);
    xGrid
      .enter()
      .append("path")
      .attr("class", "d3-3d grid")
      .merge(xGrid)
      .attr("stroke", "black")
      .attr("stroke-width", 0.3)
      .attr("fill", (d) => (d.ccw ? "#eee" : "#aaa"))
      .attr("fill-opacity", 0.7)
      .attr("d", grid3d.draw);
    xGrid.exit().remove();
    /* ----------- POINTS ----------- */
    const points = svg.selectAll("circle").data(data[1], key);
    function GetColor(x, y){
      // console.log("x: %d, y: %d", x, y);
      // return (x > 0) ? 5 : -5 + (y > 0) ? 3 : -3;
      if (x >= 0 && y >= 0) return schemePastel1[0];
      if (x < 0 && y >= 0) return schemePastel1[1];
      if (x < 0 && y < 0) return schemePastel1[2];
      if (x >= 0 && y < 0) return schemePastel1[3];
    }
    if(recolor){
      points
      .enter()
      .append("circle")
      .attr("class", "d3-3d")
      .attr("opacity", 0)
      .attr("cx", posPointX)
      .attr("cy", posPointY)
      .merge(points)
      .transition()
      .duration(tt)
      .attr("r", 3)
      .attr("stroke", (d) => color(colorScale(d.id)).darker(3))
      .attr("fill", (d) => GetColor(d.projected.x - origin.x, d.projected.y - origin.y))
      .attr("opacity", 1)
      .attr("cx", posPointX)
      .attr("cy", posPointY);
    }else{
      points
      .enter()
      .append("circle")
      .attr("class", "d3-3d")
      .attr("opacity", 0)
      .attr("cx", posPointX)
      .attr("cy", posPointY)
      .merge(points)
      .transition()
      .duration(tt)
      .attr("r", 3)
      .attr("stroke", (d) => color(colorScale(d.id)).darker(3))
      .attr("opacity", 1)
      .attr("cx", posPointX)
      .attr("cy", posPointY);
    }
    points.exit().remove();
    /* ----------- x-Scale ----------- */
    const xScale = svg.selectAll("path.xScale").data(data[3]);
    xScale
      .enter()
      .append("path")
      .attr("class", "d3-3d xScale")
      .merge(xScale)
      .attr("stroke", "black")
      .attr("stroke-width", 1.5)
      .attr("d", xScale3d.draw);
    xScale.exit().remove();
    /* ----------- y-Scale ----------- */
    const yScale = svg.selectAll("path.yScale").data(data[2]);
    yScale
      .enter()
      .append("path")
      .attr("class", "d3-3d yScale")
      .merge(yScale)
      .attr("stroke", "black")
      .attr("stroke-width", 1.5)
      .attr("d", yScale3d.draw);
    yScale.exit().remove();
    /* ----------- z-Scale ----------- */
    const zScale = svg.selectAll("path.zScale").data(data[4]);
    zScale
      .enter()
      .append("path")
      .attr("class", "d3-3d zScale")
      .merge(zScale)
      .attr("stroke", "black")
      .attr("stroke-width", 1.5)
      .attr("d", zScale3d.draw);
    zScale.exit().remove();
    /* ----------- y-Scale Text ----------- */
    const yText = svg.selectAll("text.yText").data(data[2][0]);
    function GetYText(y){
      if (y==-11){
        return "[Arithmetic Intensity]";
      }else{
        return  (-y*10 + 100)/2+"%";
      }
    }
    function GetYWeight(y){
      if (y==-11){
        return 700;
      }else{
        return  350;
      }
    }
    yText
      .enter()
      .append("text")
      .attr("class", "d3-3d yText")
      .attr("font-family", "system-ui, sans-serif")
      .merge(yText)
      .each(function (d) {
        d.centroid = { x: d.rotated.x, y: d.rotated.y, z: d.rotated.z };
      })
      .attr("x", (d) => d.projected.x)
      .attr("y", (d) => d.projected.y)
      .style("font-weight", (d) => GetYWeight(d.y))
      .text((d) => GetYText(d.y))
      .attr("fill", "#78E2A0");
    yText.exit().remove();
    /* ----------- x-Scale Text ----------- */
    const xText = svg.selectAll("text.xText").data(data[3][0]);
    xText
      .enter()
      .append("text")
      .attr("class", "d3-3d xText")
      .attr("font-family", "system-ui, sans-serif")
      .merge(xText)
      .each(function (d) {
        d.centroid = { x: d.rotated.x, y: d.rotated.y, z: d.rotated.z };
      })
      .attr("x", (d) => d.projected.x)
      .attr("y", (d) => d.projected.y)
      .attr("z", (d) => d.projected.z)
      .text((d) =>  d.x == 10 ? "[Hardware Enablement]" : "")
      .style("font-weight", 700)
      .attr("fill", "#78E2A0");
    xText.exit().remove();
    /* ----------- x-Scale Text ----------- */
    const zText = svg.selectAll("text.zText").data(data[4][0]);
    zText
      .enter()
      .append("text")
      .attr("class", "d3-3d zText")
      .attr("font-family", "system-ui, sans-serif")
      .merge(zText)
      .each(function (d) {
        d.centroid = { x: d.rotated.x, y: d.rotated.y, z: d.rotated.z };
      })
      .attr("x", (d) => d.projected.x)
      .attr("y", (d) => d.projected.y)
      .attr("z", (d) => d.projected.z)
      .text((d) =>  d.z == 10 ? "[Work Reduction]" : "")
      .style("font-weight", 700)
      .attr("fill", "#78E2A0");
    zText.exit().remove(); 
    selectAll(".d3-3d").sort(points3d.sort);
  }
  function posPointX(d) {
    return d.projected.x;
  }
  function posPointY(d) {
    return d.projected.y;
  }
  function init() {
    xGrid = [];
    scatter = [];
    yLine = [];
    xLine = [];
    zLine = [];
    let cnt = 0; 
    for (let z = -j; z < j; z++) {
      for (let x = -j; x < j; x++) {
        xGrid.push({ x: x, y: 0, z: z}); // grid position
        scatter.push({
          x: x,
          y: randomNormal(0, 0.8)()*3,
          // y: randomUniform(9, -9)(),
          z: z,
          id: "point-" + cnt++,
        });
      }
    }
    range(-10, 12, 1).forEach((d) => {
      yLine.push({ x: 0, y: -d, z: 0 });
      xLine.push({ x: -d, y: 0, z: 0 });
      zLine.push({ x: 0, y: 0, z: -d });
    });
    const data = [
      grid3d(xGrid),
      points3d(scatter),
      yScale3d([yLine]),
      xScale3d([xLine]),
      zScale3d([zLine]),
    ];
    processData(data, 1000, true);
  }
  function dragStart(event) {
    mx = event.x;
    my = event.y;
  }
  function dragged(event) {
    beta = (event.x - mx + mouseX) * (Math.PI / offset);
    alpha = (event.y - my + mouseY) * (Math.PI / offset) * -1;
    const data = [
      grid3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(xGrid),
      points3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)(scatter),
      yScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([yLine]),
      xScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([xLine]),
      zScale3d.rotateY(beta + startAngle).rotateX(alpha - startAngle)([zLine]),
    ];
    processData(data, 0, false);
  }
  function dragEnd(event) {
    mouseX = event.x - mx + mouseX;
    mouseY = event.y - my + mouseY;
  }
  selectAll("button").on("click", init);
  init();
});
</script>
