# 有狀態分佈式系統分類學

> 本文討論了CAP Theorem的侷限性，梳理了基於一致性、可用性這兩個理想屬性間的權衡的更細緻精確的有狀態分佈式系統分類學。

---

LLMS index: [llms.txt](/llms.txt)

---

## CAP theorem的討論範圍是狹窄的
在分佈式系統領域，CAP theorem廣泛被引用，而且經常被用以指導超出它討論邊界的問題。

被形式化證明（見[Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-​Tolerant Web Services](https://users.ece.cmu.edu/~adrian/731-sp04/readings/GL-cap.pdf)）的CAP theorem其實只侷限在read-write storage場景：一個只包含get、set(x)兩種操作的存儲系統，這種系統被稱爲register。

在異步網絡（指的是消息傳遞時間無界）中實現register是無法同時滿足下列屬性的：
1. Availability：所有發往register的請求最終都能完成。這和大多數真實系統的定義是不同的，因爲真實系統不要求100%完成請求，只要保證SLA夠高，同時又往往有一定時間約束，超時就會返回timeout錯誤。
2. Consistency：所有讀、寫操作均是linearizable的：B操作在A操作之後成功執行，則B看到的系統狀態不能比A完成時的系統狀態更舊。
3. Partition tolerance：允許網絡丟包。

Partition tolerance默認是滿足的，因此CAP-availability、CAP-consistency兩種屬性二選一。
離開了形式化證明的場景後，CAP theorem還有指導意義嗎？答案是否定的——除非重新定義availability和consistency，將其推廣到更通用的場景：從單對象單操作到多對象多操作的事務系統。
重新定義availability和consistency後的分佈式系統分類學雖然和CAP theorem非常類似，但不能稱之爲CAP theorem。


## 一致性
更通用的“一致性”應定義爲“併發系統中共享狀態更新的可見性”。
現代微處理器、分佈式系統、數據庫的共性是——它們都是存在共享數據的併發系統。
當我們討論一致性（consistency）時，可能指的是微處理器架構和系統編程領域的一致性模型，也可能指的是分佈式系統領域的副本一致性，還可能指的是數據庫領域的事務隔離性。這些領域的抽象層次不同，共同點是討論的系統都是併發系統。

一致性模型（consistency model）是用於描述微處理器架構領域的多核併發場景下，各個處理器被允許的亂序程度——亂序的約束越少，效率越高，併發程序正確性越難保證。
1. 最強的strict consistency指的是任何寫在任何時鐘週期都立刻對任何處理器可見，顯然不能推廣到分佈式系統領域。
2. 次之的sequential consistency指的是寫操作順序對於每個副本而言都是一致的，即各進程內部的program order一致，而不同進程執行的順序可以不一致。這個概念最初也是Lamport在討論multi-processor computer如何正確執行併發程序時提出的。和分佈式系統的副本一致性無關。C++中std::memory_order_seq_cst即可保證線程內部的program order。
3. 更寬鬆的causal consistency指的是寫操作中有依賴關係的那一部分的順序是一致的，即各進程中的dependency order一致。現代CPU基本上都是out-of-order流水線，在保證dependency order這個底線後，能多亂序就多亂序。在C++中用std::memory_order_consume的load(A)和std::memory_order_acquire的store(B)配合，即可保證這個store之前所有寫操作中load(A)依賴的那一部分對load(A)是可見的。如果每個依賴都保證Release-Consume ordering，則依賴鏈就有序，整體上可滿足causal consistency。
4. 除了上述幾個著名模型外，還有幾十個不同方法、領域中應用的一致性模型，下圖就包含了非事務分佈式存儲系統中種種一致性模型（詳見[Consistency in Non-Transactional Distributed Storage Systems](https://arxiv.org/pdf/1512.00168.pdf)）。

![Consistency1](https://wujipeng.com/img/1.png)

併發程序顯然可以很容易推廣到分佈式複製狀態機，只是增加了網絡延遲。因此，一致性模型可以推廣應用到分佈式系統中的副本一致性。以sequential consistency爲例，增加了實時約束後就是分佈式系統領域中更被廣泛引用的linearizability，指的是單個被複制對象上的單個操作滿足：A是寫操作，B是對副本的讀，A happened-before（因果律上的先於，見https://en.wikipedia.org/wiki/Happened-before） B，則A的寫對B的讀總是可見。比較一下C++的sequentially consistent ordering定義：everything that happened-before a store in one thread becomes a visible side effect in the thread that did a load。二者是一致的。

必須注意，迄今爲止討論的對象僅限單個被複制到不同副本中的對象上的單個操作，分佈式存儲不可能只存一個對象，有很多分佈式存儲支持事務或BatchWrite，涉及到多個對象上的多個操作。將單對象上單操作的可見性推廣到多對象上多操作也不困難——事務的ACID隔離級別本質上就是將單個共享對象上單個操作的可見性推廣到多個對象上一組操作上。下圖的左側就是數據庫領域熟知的隔離級別，和分佈式系統、微處理器架構、多核編程一樣，亂序的約束越少，效率越高，併發程序正確性越難保證。

![Consistency2](https://wujipeng.com/img/2.png)


## 可用性
更通用的“可用性”應定義爲“在施加某種約束後，系統仍最終能響應所有請求，無論網絡分區持續多久”。
比CAP theorem更完善的分佈式系統分類學可參考[Highly Available Transactions: Virtues and Limitations](https://arxiv.org/pdf/1302.0309.pdf)。
這篇論文裏給出了新的可用性定義：

1. High availability：每個用戶向運行中的系統發的請求，最終都會得到回覆，無論網絡分區持續多久。這就是CAP-availability，或者說traditional availability的標準定義。
2. Sticky availability：每當用戶事務在一個數據庫狀態（該狀態反應了之前該用戶所有操作）拷貝上執行，最終都會得到回覆，無論網絡分區持續多久。 這比CAP-availability要求更高。
  - 只追求high availability，用戶可以訪問系統中的任何一個replica，不同操作在不同replica上響應也沒關係；但若追求sticky availability，用戶則需要保證連續的若干操作總是在同一個replica上。比如Dynamo這種multi-writer的分佈式存儲，不能寫一會兒A節點，再寫一會兒B節點。
3. Transactional availability：分佈式系統文獻的一致性模型大多考慮的都是單對象上單個操作的場景，而數據庫文獻中關注的是事務：多個對象上多個操作合起來稱爲一個事務。顯然CAP-availability定義也不適用於事務。
  - 事務的replica availability：事務能爲它需要訪問的各個對象聯繫到至少一個replica。這個要求是比CAP-availability低的。
  - 事務的liveliness：假設我們讓每個事務都abort，就可以保證100%的及時響應，完美實現CAP-Availability，但又有什麼意義呢？因此還需要保證儘可能讓事務commit，而不是abort。
  - 因此，最終給出的transactional availability定義是：對事務中每個數據都保證replica availability，並且最終能夠在N次retry內commit成功，或internal abort（由事務自己主動選擇的abort，而非系統實現將其abort）。
  - 更進一步還可以給出sticky transactional availability的定義：如果系統能保證sticky availability，則能保證transactional availability。

根據這種定義，可以將現有的事務系統的consistency（隔離級別）與其availability進行比較，得出下圖中的結果：在新的分類學中，availability要求越高，consistency要求越寬鬆是成立的。

![Availability](https://wujipeng.com/img/3.png)
