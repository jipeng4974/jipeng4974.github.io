# DPDK is All You Need

> 對於訪存密集的數據中心應用來說，DPDK提供了非常好的性能工程範式。

---

LLMS index: [llms.txt](/llms.txt)

---

對於訪存密集的數據中心應用來說，DPDK提供了非常好的性能工程範式。本文只是淺嘗輒止，彙集其中一部分值得借鑑的思想。

## EAL：用戶空間庫
EAL(Envionmemt Abstraction Layer)是DPDK面向用戶的用戶空間庫，提供了各種有用的工具，比如：運行時對CPU特性進行檢測，更適合現代硬件的內存管理，綁核和指定任務在某個核上運行。

`rte_eal_init()`是初始化EAL的函數，有多平臺的實現：Linux、FreeBSD、Windows。以它爲例，可大致瞭解DPDK EAL做了哪些工作。

`rte_eal_init()`是一個冗長的初始化過程，包含下列步驟：檢查CPU類型是否是DPDK支持的、設置日誌等級、檢測各個socket上的各個cpu、enable每個邏輯核、初始化插件（加載共享庫，比如一些PMD drivers）、初始化[tracing機制](https://doc.dpdk.org/guides/prog_guide/trace_lib.html)、解析各個設備的配置選項、初始化全局配置（主核id、邏輯核數、numa nodes數、iova模式[^1]、內存拓撲配置）、初始化中斷處理機制、初始化多進程通用channel、掃描所有總線上的設備、初始化malloc heap、註冊多進程action callbacks（熱插拔支持）、初始化巨頁信息[^9]、初始化內存和memzone、初始化HPET/TSC計時器[^8]、檢查本地socket上的內存、創建主線程和子線程的通信信道、創建工作線程、綁核、在工作線程上啓動dummy function、初始化服務、嗅探所有總線上的設備和驅動、啓動服務、開啓telemetry（提供ethdev stats、ethdev port list、eal parameters等狀態查詢）。

## 切割需要權限的工作
DPDK程序跑在用戶態的一個前提是有內核驅動幫忙處理一些硬件設備註冊、中斷映射的事情。Linux上有幾個可用的內核驅動，比如``vfio-pci``、``igb_uio``、``uio_pci_generic``。它們是泛用的PCI內核驅動模塊，對所有PCI設備均適用。``igb_uio``基於Linux UIO提供所有類型的中斷支持，比較古老，也比較簡單，不支持IOMMU，因此IOVA mode只能用PA mode。``uio_pci_generic``和``igb_uio``類似，不過不支持MSI和MSI-X中斷。``vfio-pci``支持基於IOMMU做IOVA映射，兼容VA mode和PA mode，如果能用`vfio`，就用`vfio`，目前`uio`基本處於半廢棄狀態。

大多數設備需先從Linux內核驅動上解綁，然後再綁定到DPDK的內核驅動上。需用戶在運行DPDK程序前，用`usertools`目錄下的``dpdk-devbind.py``腳本做好設備和內核模塊的解綁和綁定——這種需要root權限的準備工作也從用戶態庫中剝離了。

## 利用巨頁，畢竟4KB已是古老時代的殘響
DPDK基於mmap在hugetlbfs中進行巨頁物理內存申請。使用更大的內存頁相比4KB默認頁(在x86上，DPDK目前支持2MB或1GB的巨頁[^7])，所需的page table entry總數大大減少，可顯著減小page table size和tlb size、降低tlb miss和page table walk開銷，提升內存分配的連續性和內存訪問的局部性，這些都有助於提升內存帶寬。

## 尊重NUMA Node拓撲
DPDK的每個操作都是NUMA-aware的，提供的API默認是NUMA node親和的，這讓用戶很難寫出遠端內存訪問的代碼。

## 尊重內存的硬件拓撲
DPDK的內存分配做得非常精細，利用了內存硬件拓撲等內存配置。

內存配置中有兩個影響應用層的重要概念：內存通道（memory channels）、內存列（memory ranks）。

內存通道是CPU和內存之間的通信通道，理論上來講內存帶寬和通道數成正比，單個通道位寬64bit，2個就是128bit。內存通道數往往等於單socket支持的DIMM[^3]數，畢竟足夠多的內存還需足夠多的通道才能保證和CPU的互連。

CPU和內存之間是64bit接口，但單個內存顆粒（DRAM chips）的位寬可能是4bit、16bit，需要多個內存顆粒並聯形成一個64bit的內存列，連接到同一組chip select上，從而保證各內存顆粒可被同時訪問。內存模組必須至少形成一個內存列，才能和CPU通信。內存標籤上的2R×8就是指列數爲2，顆粒位寬8bit，一共16個顆粒。

> Depending on memory configuration on x86 arch, objects addresses are spread between channels and ranks in RAM

x86架構下內存通道和內存列在內存地址上interleaving，即均勻分佈且遞增，因此RAM可視作由$n_{chan}\times n_{rank}$個block組成的，其DIMM架構如下圖。
![2chan4rank](https://wujipeng.com/img/2chan4rank.svg)

如圖所示，內存池最好不要讓對象的起始地址反覆命中同一個channel或同一個rank，而是要充分利用不同內存通道、不同內存列，避免通道、列之間的負載不均，提升訪存帶寬。

DPDK的`mempool`給對象大小加恰當的padding，令內存池中的下一個對象的起始地址分佈在不同內存通道和列中。具體實現可參考下面的代碼，其中64B[^4]是x86的`cache line size`，也恰恰是一個`block size`，或`memory bus width`、`channel width`。無論如何，內存池裏的地址都要首先保證是64B的整數倍，能整齊地放入cache，做到cache-friendly——事實上也是block friendly、memory bus width friendly，然後纔是保證下一個對象的`block id`和$n_{chan}\times n_{rank}$互質。

```C 
static unsigned int arch_mem_object_align(unsigned int obj_size)
{
	unsigned nchan = rte_memory_get_nchannel();
	unsigned nrank = rte_memory_get_nrank();
	unsigned new_obj_size = (obj_size + 63) / 64; 
	while (get_gcd(new_obj_size, nrank * nchan) != 1)
		new_obj_size++;
	return new_obj_size * 64; 
}
```

## IOVA和VA模式的連續性
硬件不知VA，用戶空間不知PA，DPDK的作用之一是bridge物理地址（PA）和虛擬地址（VA），因此給出一種IOVA(IO Virtual Address)是自然而然的設計。

DPDK的IOVA模式有兩種：PA mode和VA mode。如果用了PA mode，則分配給DPDK的所有IOVA地址都是物理地址，或者說，也是IO虛擬地址，只不過這個IO虛擬地址的內存佈局與物理地址完全相同。PA mode的缺點是需要root權限以讀取頁表，且可能會繼承物理內存的碎片性。因此DPDK引入了新的VA mode，一方面無需root權限，另一方面基於IOMMU[^2]做物理內存的重映射，保證IO虛擬地址的連續性，並使其編碼佈局和一般虛擬地址在格式上做到匹配，這樣就允許大片連續IOVA內存的申請。無論是硬件視角下，還是用戶空間視角下，VA mode下IOVA內存區都是連續的。

![iova](https://wujipeng.com/img/iova.png)

## 固定物理地址剛好宜用DMA
尊重NUMA拓撲、尊重內存佈局、IOVA的VA模式、使用巨頁這些特性疊加起來，天然就決定了DPDK設計中，用戶態進程用到的所有虛擬地址的underlying物理地址都是固定不變的——也就是說，這些地址是可以用於DMA的。DPDK用戶態程序可不必涉足IO事務，讓硬件自主代勞，通過固定不變的物理地址上的DMA事務完成。

## 多進程範式
DPDK還特別爲多進程做了支持，允許一個主進程管理所有DPDK資源，多個子進程共享資源訪問權。DPDK的額外努力在於它保證了子進程視野中的地址和peer進程、主進程視野中的地址是完全一樣的，也就是說連指針都能跨進程傳遞——這聽起來就相當危險，但性能肯定比各種安全的通信協作機制強。此外，DPDK還支持跨進程的全局鎖，使多進程編程更接近多線程編程。

[^1]: [Memory in DPDK Part 2: Deep Dive into IOVA](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-2-deep-dive-into-iova.html)
[^2]: IOMMU是連接在DMA-capable IO總線和主存之間，將設備的物理地址映射到虛擬地址空間的專用硬件。物理機通常都支持IOMMU，以Intel爲例，IOMMU技術即Vt-d：Intel® Virtualization Technology for Directed I/O。
[^3]: DIMM(dual in-line memory module)，即ram stick，內存條，DDR(Double Data Rate)技術的物理具現。
[^4]: 無論是i686還是x86_64，cache size都是64B，不過有些場景下合理的cache padding size是128，因爲prefetcher一次取兩個cacheline。
[^5]: [Memory in DPDK Part 4: 18.11 and Beyond](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-4-1811-and-beyond.html)
[^6]: [Memory in DPDK Part 3: 17.11 and Earlier Releases](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-4-1811-and-beyond.html)
[^7]: [Memory in DPDK Part 1: General Concepts](https://www.intel.com/content/www/us/en/developer/articles/technical/memory-in-dpdk-part-1-general-concepts.html)
[^8]: EAL通過`mmap`從用戶空間訪問HPET內核時間計數，暴露高精度計時器接口給服務層。
[^9]: EAL用`mmap`分配巨頁物理內存，並將這些物理內存再通過內存池API暴露給服務層。
