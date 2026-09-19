# 內存瓶頸應用的eBPF Tracing

> 介紹eBPF——前沿的Linux系統的可觀測性技術，以及基於eBPF的off-CPU性能分析。

---

LLMS index: [llms.txt](/llms.txt)

---

## 現代workloads的瓶頸
如今使用CPU而不將計算offload到GPU等加速器的數據中心應用，大多是訪存密集應用，例如倒排檢索、向量檢索、模型推理、分析型數據庫。

純粹instruction-bound的workload反而罕見。即使是最典型的計算密集場景，充滿了矩陣乘加的大模型訓推，實際瓶頸也出現在內存IO、off-package互連上。

現代硬件的發展趨勢是越來越深的內存hierarchy，相應地，現代workloads（訓推、檢索、數據分析）的瓶頸也逐漸向內存總線、Cache、MMU、CPU-to-CPU互連、inter-node互連轉移。

如今內存慢得就像過去的磁盤。Rust的hashmap用B樹而非紅黑樹實現。ScyllaDB和DragonFly通過利用numa機器局部性以及更好的cache優化分別擊敗Cassandra和Redis。至少，對不自研加速器硬件的互聯網公司的研發團隊——即數據中心應用的開發者來說，矩陣乘加offloading到GPU/ASIC/AMX即可，訪存優化爲代表的互連優化纔是性能工程的主戰場。


## CPU Profiling和CPU利用率的侷限性
對於訪存密集應用來說，很多人被CPU利用率高誤導，以爲計算是瓶頸，於是做CPU profiling[^1]和計算優化，往往效果不佳。假設只有10%線程時間在CPU上跑，而70%在等內存讀寫，那無論CPU profiling做得多好，也沒辦法找到真正的瓶頸。```perf```是典型的CPU profiling工具，```perf record -F 1000```是按照1000Hz採樣，對各個函數上的時鐘週期可以給出比較準確的估計，但這種採樣在阻塞階段不生效。 類似地，CPU火焰圖也只是給出不阻塞時的採樣數據，CPU火焰圖的最上沿的函數就是所有on—CPU函數，其耗時之和是CPU time之和，不包括off-CPU time。

CPU利用率在目前仍然被廣泛使用，但它已經過時。CPU利用率的真實含義是“非空閒率”，即CPU沒有跑idle thread的比例，也就包括了內存IO阻塞、網絡IO阻塞、spinlock盲等。這個概念過於古老，它出現時尚不存在memory wall，cpu並不顯著快於主存，這顯然已經不適用於現代硬件語境，容易給人計算單元是系統瓶頸的錯覺。

## eBPF off-CPU分析：in-kernel簡報 
怎麼度量off-CPU time？最簡單的辦法是應用層tracing，記錄重點代碼各個函數和代碼塊的耗時。大多數情況下，這其實就是最好的方案，精度也不錯。不過相比火焰圖還不夠帥，也不夠全面，畢竟在哪加時間戳依賴主觀判斷，有時候真正的瓶頸會出現在意想不到的地方。

Linux 4.8+[^3]可使用eBPF做off-CPU分析。比如eBPF工具[bcc/cpudist](https://github.com/iovisor/bcc/blob/master/tools/cpudist.py)，[bcc/offcputime](https://github.com/iovisor/bcc/blob/master/tools/offcputime_example.txt)。offcputime生成的call stacks可以直接用[flamegraph.pl](https://github.com/brendangregg/FlameGraph)繪製off-CPU火焰圖。bcc包含了大量工具，其具體使用方式可參照其[官方tutorial](https://github.com/iovisor/bcc/blob/master/docs/tutorial.md)。

eBPF tracing與傳統的off-CPU tracer(比如```perf```)相比，最顯著的優勢是不必把所有內核事件往用戶空間dump（調度事件是非常頻繁的，```perf```往往生成巨量數據，注入的額外開銷太大，不僅僅是CPU開銷，還有磁盤IO開銷），而是在內核就按照某種可編程的規則做了總結，把精簡的信息輸出出來。

此外，內核支持eBPF後，各種off-CPU分析需求都可以統一用eBPF實現，不再需要針對不同場景使用或製作不同的工具了——此前用```perf```做事件追蹤，用storage tracing做存儲IO追蹤，用內核統計數據觀測調度時延，不成體系，且性能良莠不一。

eBPF追蹤off-CPU時長的思路是在context switch事件結束時記錄一次stack（off—CPU期間stack是不變的，一次足矣），爲當前context的off-CPU時長增加線程睡眠時間。其僞代碼如下：
```python
on context switch finish:
	sleeptime[prev_thread_id] = timestamp
	if !sleeptime[thread_id]
		return
	delta = timestamp - sleeptime[thread_id]
	totaltime[pid, execname, user stack, kernel stack] += delta
	sleeptime[thread_id] = 0

on tracer exit:
	for each key in totaltime:
		print key
		print totaltime[key]
```

## 所以，什麼是eBPF？
簡單說，eBPF是一個允許跑自定義代碼做一些tracing和系統監控的in-kernel runtime，是BPF的升級版。

![ebpf](https://wujipeng.com/img/ebpf.png)

最初的BPF，即Berkeley Packet Filter，是一個用於報文過濾的幾乎被遺忘的古老內核特性。eBPF在BPF基礎上做了擴展，允許事件源從報文擴展到多種多樣的事件源，eBPF VM有更大的存儲空間，更多寄存器和64位word size——BPF事實上提供了一個in-kernel的沙盒環境，或者說虛擬機，安全且受限地執行用戶定義的程序。因此eBPF機制的出現實際上在內核態程序、用戶態程序之外創造了新的軟件品類。

eBPF程序不是預編譯或解釋的，而是JIT CO-RE[^2]的，程序出錯既不abort，也不panic，而是返回error message。內核態直接訪問資源，用戶態通過系統調用或fault訪問資源，eBPF則是通過一些受限的helper訪問資源——目前來說主要作用還是tracing，做一些可觀測性上的工作。

![ebpf2](https://wujipeng.com/img/ebpf2.png)

eBPF把JIT編譯器和安全驗證器直接放到了內核裏，用戶態的bpf程序先經過parser變成AST，再做一些構造和語法分析，然後生成IR，最終生成優化後的bytecode。BPF bytecode作爲輸入進入內核的JIT和verifier再編譯成機器碼給CPU執行。

![ebpf3](https://wujipeng.com/img/ebpf3.png)

## eBPF的其他應用
eBPF除了用在可觀測性上，還可以應用於網絡，在L3/L4/L7做traffic control, monitoring或load balancing，比如libbpf ```tc```/```qdiscs``` library, ```XDP```(裸金屬高性能可編程網絡)/```Cilium```(高性能雲原生網絡)/```Katran```(傳輸層負載均衡)。

此外，eBPF還可以應用於安全領域。畢竟eBPF可以觀測系統中的各種事件，比如監控某些敏感文件（/etc/passwd這種）是否被篡改。基於這種觀測能力在加上一些安全相關的先驗知識，就可以做一些安全工具。K8s的```seccomp```工具就是基於eBPF實現的。


[^1]: 區別於off-CPU分析，這裏的CPU profiling指狹義的on-CPU分析，不考慮阻塞中的thread time。
[^2]: CO-RE: Compile Once Run Everywhere，也就是說BPF bytecode是可以relocate的。
[^3]: 不過Linux 5.x纔有完整的CO-RE和BTF支持，其中BTF(BPF Type Format)是爲eBPF設計的內核數據結構描述機制。
