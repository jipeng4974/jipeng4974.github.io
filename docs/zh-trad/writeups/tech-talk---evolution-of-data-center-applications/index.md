# Tech Talk: Evolution of Data Center Applications

> Tech Talk文稿，主旨是白盒化算法理論和基礎設施，給出一個對Datacenter應用，尤其是Datacenter AI演進趨勢的主觀理解。

---

LLMS index: [llms.txt](/llms.txt)

---

數據中心應用（數據庫、消息隊列、檢索服務、參數服務器、直播服務、廣告服務等）作爲互聯網服務的主體，位於終端應用和硬件基礎設施之間，橋接了需求端和硬件端，同時還受到成本、營收、算法、法律等因素的制約。
近期硬件側、算法側均有變化或變革，因而有必要反思數據中心應用的演進方向，討論數據中心應用的近未來發展趨勢、潛在的創新點和可收割的低垂果實。

## 變革中的不變量：數據中心應用能耗
全球數據中心能耗十年來幾乎沒有增長，而同期的服務需求、計算、存儲、數據傳輸總量則經歷了數量級增長。作爲從業者，我們對互聯網的爆發式增長應該不陌生，因此全球數據中心能耗從10到18年僅漲了6%的事實略微反直覺——從能耗的角度看，這幾乎是一個停滯的產業。

CMOS工藝和軟硬件技術的整體進步可以解釋服務器能量效率上的提升，但能耗停滯的根本原因還在於互聯網企業盈利的本質是對有支付能力的10億國外人口和10億國內人口征服務稅，個別成功的初創企業可以迅速擴張，但互聯網行業作爲整體，在數據中心能耗上的投入增長受制於全球平民收入增長。

![DatacenterPower](https://wujipeng.com/img/DatacenterPower.jpeg)

當初創企業的規模擴張到互聯網規模後，機器資源的增長速率應該會從爆炸式上升陡然轉入停滯，進入自然汰換階段。
從這個角度可以得出結論，數據中心應用在規模擴張結束後，也應從能耗粗放型轉爲精細集約型，通過軟硬件協同充分釋放硬件潛力，通過算法創新減少總計算量。

## 算法迭代：從演繹推理到歸納推斷
算法側的趨勢是“transformers getting even more attention”。

計算機理論源於符號邏輯，1930年代圖靈、哥德爾、邱奇各自獨立提出圖靈機、廣義遞歸函數、lambda演算，以及三者相應的可計算性。圖靈機的停機問題和哥德爾不完備定律駁斥了1920年代希爾伯特計劃的可行性，證明了形式化系統是不完備的，通過有限個公理和規則永遠無法推導出全部的真理。

正如人的無知可以分爲問題（可計算）和神祕（不可計算），人的理性也可以分爲演繹推理和歸納推斷，前者求解問題，後者分析神祕。

計算機程序基於圖靈機模型，天然就是適合進行演繹推理的形式化系統。至今我們依然基於規則系統(人工編程進行條件控制）、信號處理（比如音頻特徵提取）、狀態機等形式化方法實現大多數數據中心應用：比如各類數據庫、檢索系統、參數服務器、遊戲AI。

AI時代來臨之後，通過更多的權重（作爲公理）和由大量算子（作爲規則）組成深度神經網絡（從實現角度來說，仍然是形式化系統）去模擬或者說近似“歸納推斷”。（純粹貝葉斯主義的形式化表達是所羅門諾夫歸納推斷法，這個方法本身是永不停機，越算越停不下來的，所以只能近似，不能用實現）這種近似無論多麼粗糙、低效，歸根到底還是提升了計算機語言和集成電路的表達能力，完成了推理到推斷的跳躍。

數據中心應用多了歸納推斷能力之後，就產生了對複雜現實（現實信號，如圖片、語料庫、音頻，整體上具有高維度的同時，局部往往還存在具有高數據精緻度的結構，這裏精緻度可以用香農信息熵或柯氏複雜性形式化定義）理解能力(pin down reality)，因此湧現出了一些基於知識的AI系統(knowledge-based AI)，比如推薦系統、廣告投送、翻唱識別、聊天機器人。共同點是將現實映射到表徵低維子空間上，這個子空間裏的向量或者說embedding可以從某個角度有效刻畫現實。Transformer做得更進一步，等價於將現實映射到多個相干性低的低維子空間上（見https://arxiv.org/abs/2306.01129），從而從多角度多層次理解現實，通過表徵空間編碼率與各子空間編碼率之和的差值度量特徵的緊湊性和判別力。

在遊戲和博弈領域，強化學習也取得了規則系統無法企及的成就，在星際2、Dota2（固定陣容5v5，solo）、圍棋等領域擊敗職業選手。

此外，歸納推斷相比演繹推理還有一個重要特質，那就是更符合人類大腦（或推廣到任意地球生物大腦，蜜蜂的3d尋路避障能力，蚊子的血管定位能力都是基於天然的貝葉斯歸納推斷。也只有受過良好訓練的人類才能進行緩慢的演繹推理）的思維習慣，因此人類社會的傳統，人類語言的不確定性、人類的偏好都難以用形式化方法描述，卻可以被概率語言描述，被歸納推斷方法預測。

算法進步的下一個課題則是研究同時具備演繹推理和歸納推斷能力的模型。現有的Transformer的確展現出簡單多步算術的擬合能力，但擬合出來的是一個步數相關的超複雜非線性系統，而不是簡單的算術規則，所以往往正確率到八十左右就再也上不去了。對於形式化系統來說，推理步數不影響公理和規則總數，只稍稍影響計算量。但對於LLM這種歸納推斷模型來說，一旦推理步數變多，計算變得更復雜，LLM的符號邏輯能力就會迅速下降，在訓練數據中從未出現過相同執行路徑的動態規劃問題裏面可以直接降到0。理論上來說，無限精度的Transformer是圖靈完備的（見https://arxiv.org/abs/1901.03429），而只要不是無限精度，就不是圖靈完備，實際應用中模型權重和計算的精度即使以常規計算機應用標準看也是非常低的，而模型架構又沒有針對多步邏輯推理進行設計，再加上深度學習本身就是在擬合一個近似函數，三重因素疊加導致現有LLM的演繹推理能力不盡如人意。

從算法迭代角度講，數據中心應用的發展趨勢是在保證“可計算性”(字面意思，非形式化定義)的前提下，追求“人性化”(align to humanity)、“完備化”（pin down reality）這兩個演繹推理手段無法達成的理想屬性。智能創作、個性化推薦、聰明的Agent（助手、遊戲NPC、機器人、自動駕駛），複雜系統預測（天氣預報、交易系統），大模型訓練（模型並行、高性能網絡、C2C互連、光學I/O、異構計算）都是有前景的方向。

而那些已經能用規則系統高效解決的問題則不必強行應用歸納推斷模型，那樣只會增加成本和錯誤率，比如音樂指紋識別、關係型數據庫、圖片旋轉變形、無損壓縮。甚至某些看似“神祕”的領域，如OOD text classification，基於表徵學習白盒化研究成果和信息論，也能給出直接的數學解，比如近期一個研究（“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors）就用14行代碼超過了百億參數的bert，（後來被發現實驗代碼有誤，實際上效果沒那麼好）這個論文裏提出用gzip等無損壓縮工具去近似文本數據的柯氏複雜性，用信息距離（原理類似馬毅提出的rate reduction）做KNN，效果非常好，在音頻領域也可以做類似的嘗試。

## 數據中心硬件基礎設施
先簡單介紹一下常見的數據中心硬件基礎設施：
- 比較老的是Cascade Lake 14nm的24核per die的機器，一個芯片24核，兩個就是48個物理核，也就是我們平時說的96核機器。也有其他套餐，最多是每個die上28核，總共112個邏輯核。
- 比較新的Ice Lake是10nm工藝一般是32核的，對應128個邏輯核，最貴的套餐可以到40核。仍然是基於Monolithic die，巨大的晶粒上用mesh總線把40個核放到一起。
- 最新的7nm工藝Sapphire Rapids，以及對標它的AMD Genoa今年已經量產。
- 關於網卡，現在用的比較多的是Mellanox 25G  CX4/CX5卡，也有相當多的100G dual-port CX6卡，都是支持RDMA的，不用RDMA也能提供相當好的高速以太網性能。如果是做虛擬化的，比如AWS，阿里雲，還可以把這些網卡的smartNiC能力利用起來，offload虛擬化開銷。

相對之前的Lakes，Sapphire Rapids變化非常大，此前的monolithic die路線確實走到了極限，Sapphire Rapids也進入了multi-die時代：芯片內部分爲4個die，這也可以視作向chiplet方向演化。

Sapphire Rapids最多支持8個socket，每個芯片可以支持60核。IO技術上支持CXL 1.1、PCIe 5.0、UPI 2.0，HBM2e(optional)。

![Xeon](https://wujipeng.com/img/4th_xeon.jpeg)

AMD的第四代EPYC的主力型號Genoa今年Q1也開始量產，基本上和Sapphire Rapids對標，採用5nm工藝。雖然說AVX512被很多人詬病，實用表現非常一般，但Zen4也支持了，畢竟XeongenEPYC目標客戶就是音視頻處理和AI workload越來越多的數據中心應用。Genoa是每個die/chiplet 8核的，12個CPU cluster die中間圍繞一個IO die的設計，相當於一個芯片96核，是典型的chiplet範式設計。

## 芯片設計的迭代：Chipletization
近期芯片設計領域一個顯著變革是Chiplets+SiP(System in Package)範式取代die size較大的SoC+PCB合封。
Chiplets同時受到業界和學術界的關注，被IBM research稱爲"what’s next in computing"，後續章節中對計算、IO、內存等技術的討論也都涉及chiplets和co-packaging，因此我們首先討論芯片層次的迭代。

所謂chiplet partitioning就是將電路切分成模塊化的子系統，每個子系統都是一個獨立晶粒（die），即chiplet，多個chiplet用2.5D/3D技術封裝成一個芯片(package)。

![Chiplet](https://wujipeng.com/img/chiplet.png)

Chiplet-reuse範式相比傳統的IP-reuse（IP在芯片語境下指的是具有獨立功能和成熟設計的電路模塊）的優勢如下：
1. 先進CMOS製程（7nm以下）由於技術原因不太可能在大晶粒上獲得高yield，die size越小成本越低。
2. 先進CMOS製程下，不太可能同時縮小電源管理、快速IO SerDes等模擬IP，先進CMOS一般只用於處理器和加速器。
3. 允許模塊化設計，讓設計者可以專注於單個模塊的極致優化，並選擇最合適的技術：比如CPU和GPU用先進製程，模擬模塊用成熟製程，高帶寬內存HBM用DRAM，AI加速器可以用非易失性內存。
4. 允許芯片/package層次的異構集成：讓通用CPU、優化後的GPU、嵌入式的FPGA、專用的機器學習電路、光學IO模組、高帶寬內存等模塊以合適的方式，用先進的使用硅通孔（TSV）、微凸塊（micro-bumps）、甚至die-to-wafer混合鍵合技術的3D封裝方案，像樂高積木一樣搭出完整的系統。

過去我們設想的各類DSA、AI芯片、FPGA百花齊放的異構計算時代並沒有如期到來，而是被NV的GPU軟硬件結合且計算互連一體化的方案碾軋了，幾乎只剩下TPU在繼續向v5迭代。

但未來的服務器芯片本身就存在多樣化異構共封裝集成的可能性和傾向性。CPO(co-packaged optics)、HBM(high-bandwidth memory)這些神奇物種因Chipletization的契機而得以進駐其中。

更多的功能也就意味着更高的可編程性。有些功能甚至可以帶來革命性變革，比如光學IO帶來的超高通信帶寬，HBM帶來的超高訪存帶寬，高性能offpackge互連技術（Nvlink-C2C）、多個Chiplet之間的Mesh互連(NvSwitch)，現在被Nvidia用來搭建H100，被谷歌用來搭建TPUv4，未來則可能顛覆host-centric的數據中心應用設計範式，迎來硬件資源解聚（disaggregation）的新計算體系：適應資源解聚的操作系統(LegoOS就是基於早期IB network的一個嘗試)、系統語言ABI、新的高級語言、新的網絡IO、存儲和計算形態都有可能從中孵化而生。

## 計算和內存層次的迭代：可擴展的衆核NUMA架構 
在商用服務器領域，Chiplet範式中的一部分設想已經實現了，比如AMD很早就開始應用chiplet，也部分解決了chiplet間IO問題，實現了有可擴展性的衆核NUMA架構。SPR之前的Xeon物理機也是NUMA，雖然只有2個NUMA節點（目前Intel的NUMA node太大了，所以不太好稱之爲Chiplet）。

μArch對計算/訪存密集型數據中心應用的性能工程有直接影響。下圖是一個6chiplet封裝的96核概念機。顯然當我們把集成電路的黑盒拆開，就可以看到更細粒度的組件以及它們組成的網絡（Network-on-Chip）結構。這個概念機集成了各種先進設計，不僅有many-core，還支持完整的cache coherency。相比過去的多核架構，衆核架構的內存層次也相應變得更深，cache miss的代價變得更高。以至於Rust的標準庫用B樹去實現map（而C++中衆所周知是紅黑樹），這就是處理器和內存頻率差距逐漸拉大的結果，（誇張地說）現在的內存已經慢得像是當年的磁盤了。

![IntAct](https://wujipeng.com/img/IntAct.png)

針對NUMA架構，系統層的Linux內核和KVM的NUMA-aware scheduler，應用層的網絡框架Seastar、數據庫ScyllaDB、內存數據庫DragonFly等都已經注意到感知硬件拓撲能極大提升整體性能（ScyllaDB、DragonFly分別數倍領先於對標的Cassandra、Redis），提出了share-nothing高性能架構：避免鎖和不必要的共享內存、避免不必要的遠端內存訪問、避免不必要的跨晶粒通信，設計緩存友好的數據結構，更好地利用晶粒內本地的L1 cache——考慮到目前我們用的Cascade Lake機器並非完全cache coherent，未來即使做到完全cache coherent，shared cache的coherency機制也幾乎一定有開銷，總之在複雜拓撲深內存層次時代，需警惕cache miss。

![NUMA](https://wujipeng.com/img/NUMA.png)

## 重新遇到I/O瓶頸：先進互連再次成爲HPC核心
數據中心應用的IO佔了全球IO traffic的76%，和計算一樣，IO也是耗電的，而且和計算一樣，數據中心IO耗電也十年沒有變過，被硬件進步offset掉了。和計算一樣，互連是分層級的，近期die-to-die(on-package)鏈路層面有UCIe標準的發佈，off-package層面有基於PCIe6.0的CXL3.0，900GB/s的NvLinkC2C，inter-node層面有Infiniband NDR。這些是基於電的互連，相比而言光學互連更有前景，但也更困難，而且還在早期研發階段。

過去的大數據和前大模型AI時代對IO的需求較低，標準以太網足以支撐大部分數據中心應用，包括parameter servers。大模型訓練產生了新的計算、IO形態，內存放不下模型，不得不做模型並行後，IO就重新成了瓶頸：H100的8個GPU每個都需要7.2Tbps的off-package帶寬，相比之下，連ToR交換機都只需要10+Tbps。AI專用GPU在大模型訓練場景下的帶寬需求已經非常接近交換機（交換機和GPU一樣，都是巨型ASIC，也都是co-packaged optics適用的領域）。在交換機領域，谷歌已經研發出了實用且收益顯著的純光學鏈路交換機。在GPU互連上，NV也提出過光學互連的GPU的概念系統，甚至還設計了相應的帶外置激光源的GPU機架和順便解決冷卻問題的稀疏佈線。

![optics](https://wujipeng.com/img/optics.png)

先進IO技術與HPC(高性能計算)的發展密可不分，儘管HPC或者說超算在大衆的想像中一直和超強悍的處理器、加速器直接相關，但實際上恰恰相反，傳統的HPC workload（建模、模擬類的科學計算）的計算用的往往是普通的商用節點，反而是互連必須用高性能的HPC interconnect技術。傳統超算的異構性體現在IO技術，而非FPGA、專用ASIC的應用。

後來到了大數據分析和AI時代，標準以太網足以支持適應當時模型參數量的AI訓練負載。主流的互聯網大數據應用可以完全基於商用IO技術和商用計算節點實現。而少數AI DC的異構性主要體現在加速器技術（GPU、TPU、專用AI芯片）而非IO。

如今出現了大模型訓練主導的異構負載，大模型參數量激增導致內存不足，不得不進行模型並行後，die-to-die帶寬、off-package帶寬、Inter-node帶寬重新成爲瓶頸。先進(異構)互連技術重新成爲HPC的核心話題。

Datacenter AI重回異構IO+異構計算架構，本質是超算化，因此剛好也能適應建模+模擬類的傳統HPC負載，其實給了互聯網行業一個新的機會，那就是卷贏大模型訓練的同時，還可以順便進軍超算行業，爲高校、科研機構提供廉價、可靠、易用、隨時oncall的科學計算能力，輿論上一定程度上扭轉互聯網公司對社會缺乏貢獻的負面形象，爲繼續徵收互聯網服務稅尋求合法性支撐。

## I/O技術的迭代：光學IO愈發接近計算端點
先進銅纜互連是現在，共封裝光學互連則是未來。

前文提到的Co-packaging是先進互連的關鍵技術，一方面將多個晶粒共封裝本身就可以縮短IO鏈路，降低IO能耗，另一方面允許集成共封裝光學模組技術CPO(co-packaged optics)。

數據中心IO的一個演化趨勢是"bring fiber closer to endpoints"。光鏈路相比電鏈路，一個明顯優勢是傳輸距離更遠（受制於頻率相關的衰減）。另一個優勢是隨着帶寬增加，電信號不斷變短，噪音不斷變大，IB network已經逼近銅纜極限，繼續發展下去只能從銅纜走向光纖。此外，高頻下電互連和連接器既要接收又要發射，會經歷顯著的串擾，這也限制了電互連的封裝密度。光纖作爲信號傳輸介質幾乎是理想的，唯一低效的地方就是兩端的電光轉換部分。

現在in-racks連接主流方案是銅纜，inter-rack交換則基於以太網鏈路。超大數據中心裏，纜線長就達到幾公里，因此越來越多使用光纜——甚至短距離鏈路現在也越來越多地用光纜。數據中心裏，fiber越來越接近endpoints，越來越接近cpus、gpus，最新的趨勢是直接將光學組件集成到硅片上。CPO把光電鏈路結合在一起，無需intervening receive and re-transmit的過程，把光電轉換(optoelectonic conversion)步驟省略了。第一代CPO是pluggable optics，第二代是On-Board Optics/Near Package Optics，第三代是2.5D CPO，第四代是3D CPO，第五代則是Integrated Laser。

Google的TPUv4超算最大的創新就是4k節點上可重配置的純光學鏈路的光學交換機(OCS)，節省了光電轉換的能耗， Infiniband將銅纜高性能互連發展到極致，後續的roadmap也是從銅纜到光電共封裝。Nvidia雖然一直在推電鏈路方案（Nvlink），但也和Ayar Labs簽署了研發合作關係，開始支持帶外激光器和硅光子互連技術的研究，畢竟NvLink本質還是NUMA，可以擴展到8GPU，16GPU，但不可能把數據中心規模的一萬個GPU連起來。HP也於去年與Ayar Labs合作，試圖將硅光子學引入它們的先進HPC IO產品Slignshot互連。Intel也在研究激光器嵌入芯片內部的集成方案。

下圖列出了interposer、PCB、CPO、電纜、有源光纜的耗電、成本、密度、傳輸距離指標。CPO的優勢是顯而易見的。

![CPO](https://wujipeng.com/img/CPO.png)

在當前的技術水平下，CPO僅被視爲一個電光(E/O)橋的角色，以解決SiP的互連帶寬密度瓶頸問題。對分佈式訓練等應用場景來說，電光橋，或者說bring fiber closer to endpoints已經可以大幅降低能耗，提升性能了。但CPO的潛力遠不限於此，如果給CPO chiplet稍微加一些功能，就可以像協處理器、smartNiC那樣offload一些CPU work，比如做一些簡單的數據預處理、後處理，又如CPO無需經過CPU直接就能訪問HBM，從而提供DMA能力，這對解聚架構非常有幫助，無需物理上做pooling，又比銅纜IB網絡更快。

這意味着光學IO不僅可以解決大模型訓練帶來的帶寬問題，還給數據中心應用從host-centric向解聚（disaggregated）架構轉型提供了可能。

何謂解聚範式？與傳統的服務器中心範式相反，解聚範式是指將作爲整體的服務器掰開，拆成CPU、DRAM、磁盤、加速器等獨立的硬件資源進行資源抽象和管理的數據中心應用架構設計範式。硬件解聚並非新概念。18年的USENIX OSDI最佳論文LegoOS，一句"We believe that datacenters should break monolithic servers"，充滿了信念感。當年的Infiniband還沒有進化到NDR版本，光學I/O也還遠離數據中心內部端點，但已經足以支撐這樣的宏大敘事。

有了高性能網絡，解聚架構就能有效提升數據中心應用的資源利用率，減輕物理機上CPU、加速器、內存、磁盤等資源在host-centric範式下不可避免的over-provisioning問題。

## 評估 GH200 Grace Hopper Superchip 
NVIDIA宣稱Grace Hopper Superchip是世界上第一個真正支持HPC和AI負載的異構加速平臺。
![GraceHopper](https://wujipeng.com/img/GraceHopper.png)
如下圖所示，這個superchip是一個把Grace Arm Neoverse CPU+LPDDR5x內存和H100 Tensor Core GPU+HBM，NVLink-C2C集成合封成PCB的集成方案。
![GraceHopper](https://wujipeng.com/img/GraceHopper2.png)
![GraceHopper](https://wujipeng.com/img/GraceHopper3.png)

這並不是一個創新方案，一方面悖逆Chipletization的潮流（除了HBM算是chiplet外，H100、Grace、NvSwitch都是巨型SoC/ASIC，況且即使是HBM也是PCB合封，而不是SiP），另一方面也沒有進行任何CPU-GPU超融合（物理上融合至單個SoC上，邏輯上統一頁表管理、內存、緩存、併發模型等）的探索或嘗試（GPU設計之初就存在太多和CPU無法兼容的設計，比如緩存模型、內存模型和併發模型，如今CUDA根基已成很難回頭），只是簡單粗暴的將CPU、高帶寬內存、H100以PCB合封的方式集成，用NVLink-C2C提供內存一致性和更高off-package的帶寬（並未嘗試任何先進IO技術）。軟件上也沒能在CUDA基礎上提供更強的可編程性，僅僅提供coherent memory access，編程模型仍然是完全異構的（這也是因爲CUDA自誕生之初就是個圖形加速庫，也沒法考慮未來會出現對這種superchip的同構編程模型的需求）。

但這是一個低風險高執行力的集成方案，正如扎克伯格所說，"Move fast and break nothing"，把原本優秀的組件原封不動地合封起來，不做侵入式修改，只要動作足夠快，就能迅速佔領市場，構建生態，並支撐溢價。市場上有更完美的memory coherence方案（比如AMD MI300X），更好的CPU-GPU超融合方案，也有比不得不爲圖形負載妥協的GPU效率更高的AI芯片，但就是沒有CUDA異構編程體系，以及Grace Hooper這樣把計算、內存和IO瓶頸都解決得差不多的完整解決方案。

總之，NV的方案作爲生態(GPU + CUDA)與生物(ChatGPT根據A100量體裁衣的訓練方案)互相作用下的best-of-breed，遠遠沒達到理想最優，甚至也不在正確的技術路線上，AMD的所謂APU以及國內的AI DSA（如Biren）仍有彎道超車的希望。

討論計算系統的新機會
- （應用）端到（硬件）端的全棧優化，或者說軟硬件協同。
  - TVM: deep learning compiler stack for cpu, gpu and specialized accelerators
  - GPU + CUDA
  - GH20 Grace Hopper + 新的CUDA NUMA內存API+異構編程API
  - 司內的LavaRecord全鏈路優化項目，向下(LavaUOS)對接新存儲硬件，試圖在nvme ssd上建立高效的用戶態IO軟件棧。
- 應用機器學習方法對參數空間較大的系統做auto-tuning。
  - 存儲引擎如rocksdb調參
  - 深度學習模型在異構硬件上的auto TVM
- 先進互連技術支持下的資源解聚架構設計。
  - LegoOS
  - PolarDB-X的存算分離和memory pooling
- 計算節點上的Share-nothing架構，以及data-oriented設計。
  - 應用框架層面已有Libtorque、DragonFly、Seastar、Scylladb等先例，主要是IO密集應用——不過只要是內存佔用大的CPU應用，大多可以視爲IO密集的，因爲cache miss上來之後訪存佔比往往會遠超計算。
  - 虛擬化方向，交大IPADS實驗室的CPS: A Cooperative Para-virtualized Scheduling Framework for Manycore Machines，提出協作式半虛擬化調度機制，大幅提升衆核虛擬機可擴展性。
- 基於深度模型白盒化研究和已有的數學工具，用direct math solution取代黑盒模型的近似。
  - 例如“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors用壓縮+信息距離+KNN的簡潔解決方案。
  - 用異類不相干性、同類可壓縮性（稀疏性）衡量embedding效果，不必藉助某種端到端應用的指標間接衡量。
  

## 參考文獻和進一步閱讀
- Learning One-hidden-layer Neural Networks with Landscape Design：即使是最簡單的深度學習非凸優化場景，用數學工具（數學最優化方法）進行解釋也極爲困難。
- Functionality and performance of NVLink with IBM POWER9 processors：幾年前IBM Power9（美國能源部的Summit和Sierra超算系統）就在用NVLink，而且hardware cache coherence設計（以及hardware atomic ops，addr translation）已經非常完善，比Grace Hooper方案更完善。
- Faith and Fate: Limits of Transformers on Compositionality：大語言模型湧現出演繹邏輯能力，但在多步複合問題上表現不佳，在訓練樣本中從未出現過計算圖中相同計算路徑的動態規劃問題上準確率更是迅速跌落。與其他emprical study相比，這個研究更嚴肅，也更全面，考慮了計算圖中訓練時未見的splits帶來的影響。我們有理由確信，大語言模型湧現的演繹推理能力會受制於transformer的天然侷限。
- Teaching Arithmetic to Small Transformers：基於transformer的小語言模型足以學習簡單算術能力， 提供包含正確的計算步驟的訓練數據（chain-of-thought style data）是提升算術學習能力的關鍵，簡單粗暴地用題目和結果進行訓練，單純靠增加模型大小無法提升準確率。
- A Survey of Large Language Models：提供了對大語言模型的up-to-date review。
- Variantional Inference: A Review For Statisticians：提供瞭解釋VI、理解VI的統計學家視角，討論了VI應用於指數級模型族的特例，並給出一個貝葉斯高斯混合模型的例子，並推導出一種使用隨機優化來擴展至海量數據的VI變體。
- Training language models to follow instructions with human feedback：OpenAI的經驗介紹，重點是RLHF。
- GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE：來自semianalysis的爆料，頗具可信度。
- Efficiently Scale LLM Training Across a Large GPU Cluster with Alpa and Ray：LLM訓練。
- Scaling Language Model Training to a Trillion Parameters Using Megatron：Megatron（repo： https://github.com/NVIDIA/Megatron-LM ，paper： https://arxiv.org/pdf/1909.08053.pdf ）
- https://www.youtube.com/watch?v=eqWPyaRcILQ 微軟Azure硬件系統和基礎設施團隊的Ram Huggahalli關於Co-Packaged Optics的talk。
- https://www.youtube.com/watch?v=Xt-GY8Pkt6g 研究光通信和先進互連技術的Tony Chan Carusone關於Co-Packaged Optics以及Evolution of IO的talk。
- Next-generation Co-Packaged Optics for Future Disaggregated AI Systems：對共封裝光學模組以及未來的解聚AI系統的洞察。
- TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings : Google的TPUv4，重點是純光鏈路交換機
- LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation ：樂高OS，基於早期Infiniband高速網絡做硬件資源解聚的嘗試
- https://www.hpcwire.com/2020/11/16/nvidia-mellanox-debuts-ndr-400-gigabit-infiniband-at-sc20/ Mellanox(Nvidia)的Infiniband NDR版本，以及roadmap。
- Rack-scale disaggregated cloud data centers: The dReDBox project vision: 數據中心應用解聚架構的早期嘗試。 
- White-Box Transformers via Sparse Rate Reduction：馬毅團隊對transformer的白盒化解釋，此前馬毅已經給出了更通用的rate reduction原則： Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction。
- https://github.com/bgavran/Category_Theory_Machine_Learning 深度學習的範疇論解釋。深度學習可解釋性和逆向工作還可以參考Christopher Olah的blog: colah.github.io，Olah有許多深刻的洞察，比如https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/ 流形假設的可視化和深度學習分類的解釋。
- IntAct: A 96-Core Processor With Six Chiplets 3D-Stacked on an Active Interposer With Distributed Interconnects and Integrated Power Management：先進IC設計領域的論文，給出了一個集成了chiplet範式、3d封裝、完全cache coherence等先進概念的96核衆核原型系統。
- “Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors：無損壓縮近似柯氏複雜性，然後計算信息距離（類似rate reduction，刻畫了總體與分類之間的信息差，分類的編碼長度低而總體的編碼長度高，表明這種分類具有異類強區分性和同類可壓縮性），依據信息距離做簡單的KNN即可完成分類。這個研究的代碼有錯，並不能擊敗BERT，見https://kenschutte.com/gzip-knn-paper/。
- M. Li and P.M.B. Vitányi, An Introduction to Kolmogorov Complexity and Its Applications 柯爾莫哥洛夫複雜性的介紹和應用
- A Mathematical Theory of Communication 1948年香農信息論的論文原著
- hwloc doc：hwloc的文檔，hwloc是NUMA-discovery + cpu/memory-binding library。
- https://man7.org/linux/man-pages/man2/mbind.2.html：libnuma的NUMA memory policy函數。
- On the Turing Completeness of Modern Neural Network Architectures 證明了無限精度transformer是圖靈完備的，即任意圖靈機都可被無限精度transformer模擬，但只要是固定精度就不是圖靈完備的。
