# String Lookups Reduce to Parsing

> 字符串查找和字符串解析，本質都用盡可能緊湊的結構和高效的算法，從字符流中抽取狀態。因此龍書中的NFA轉DFA算法可以派上用場。

---

LLMS index: [llms.txt](/llms.txt)

---

標題即結論：字符串查找問題可歸約爲解析問題。

這個結論源於近期一個有趣的觀察：用ragel寫高性能的ascii protocol parser，本質上是利用[nfa](https://en.wikipedia.org/wiki/Nondeterministic_finite_automata)轉[dfa](https://en.wikipedia.org/wiki/Deterministic_finite_automaton)提升性能，這和[toplingdb](https://github.com/topling/toplingdb)中將同一層各個sst對應的trie（本質是dfa）合併成一個dfa(大量dfa->nfa->1dfa)的思路是同構的。

上述同構隱隱蘊含着一個reduction：字符串查找和字符串解析，本質都用盡可能緊湊的結構和高效的算法從字符流中抽取狀態，lookups可以視作一類特殊（且模式相當規則）的parsing。LSM key lookup更是比較特殊的海量索引的key range無overlap的場景，存在大量可以輕鬆合併的DFA。因此龍書中的大量DFA轉NFA轉DFA算法可以派上用場。

KV Store的in-memory key index，可以是紅黑樹，可以是skiplist，可以是hashmap，可以說patricia trie(一種[radix tree](https://en.wikipedia.org/wiki/Radix_tree)變種)，也可以是toplingdb中的NestLoudsTrie。這種結構我們同樣可以在路由表實現中看到。字符串索引，歸根到底是字符串lookup結構。正如路由表實現可以通過把所有routes合併到一個DFA裏（很多routes都包含regex），kv數據庫也把Trie這種特殊的dfa（Trie的狀態轉移圖是樹，樹是一種無向圖，多了個任意兩結點只由一個邊連接的約束）做多索引合併，每個索引對應的key range還不重疊（LSM特性），因此合併速度非常快，合併後的DFA表示起來也簡單、緊湊，詳見[自動機算法在數據庫索引中的應用](https://zhuanlan.zhihu.com/p/628057993)，我在作者的文章下面追問了一下DFA合併的觸發條件和DFA合併開銷，作者的答覆是compaction/flush時觸發，在整個lsm更新過程中佔比很小，也不涉及多線程，無需考慮線程安全。
