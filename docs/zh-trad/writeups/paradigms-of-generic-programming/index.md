# 泛型編程的三種範式：Archetype, Ducktype, Subtype

> 本文總結泛型編程的三種範式：Archetype, Ducktype, Subtype。三者都以type結尾，一方面是因爲這樣比較帥，有規則感和邏輯上的建築美，另一方面是因爲系統語言編程本身就是在打造一個個類型，而泛型編程就是在打造一個個類型規約+遵循規約的類型。

---

LLMS index: [llms.txt](/llms.txt)

---

## 泛型不等價於模板
什麼是泛型編程？提起generic programming，很多人都會立刻想到template，但模板編程只是泛型編程的一種範式。最初發明"generic programming"這種說法的Alexander Stepanov反覆強調過：
generic programming is not about how to use template。第一個版本的STL儘管名字裏帶template，實際上是基於scheme實現的。

## 要泛型，就必須有規約
規約的制定和對規約的遵循是泛型編程的基礎。
任何泛型編程都需要一定規約，有規約才能讓一種抽象適配多個具體實體。沒有規約，那寫具體實現的人都不知道遵循什麼規範，實現什麼接口，滿足什麼條件，泛型編程自然就無從談起。

規約：在JavaScript裏是prototype；在Swift裏是protocol；在Rust裏是trait；在C++模板編程裏是concept或者SFINAE。

遵循規約：在JavaScript裏是運行時將具體對象關聯到某個原型對象；在Swift裏是用類似繼承的冒號讓具體類型conforms to某些protocol；在Rust裏是用impl SomeTrait for SomeStruct語法顯式地爲某個類型提供某個trait的實現；在C++裏模板實參無需特殊語法聲明遵循模板形參，但要心照不宣地遵循模板代碼的要求。

## 三種規約，三種範式
根據規約不同，可命名泛型編程的三種範式：Archetype, Ducktype, Subtype。
- **Archetype**：以類型的類型(type-of-type)或者說協議(protocol)、接口(interface)爲規約。
  - [Rust trait](https://doc.rust-lang.org/book/ch10-02-traits.html): "defines shared behavior"
  - [Carbon interface](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/terminology.md#interface): "defines an API that a given type can implement"
  - [Swift protocol](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/): "defines a blueprint of methods, properties, and other requirements"
  - [C++ type erasure idiom](https://davekilian.com/cpp-type-erasure.html): "captures the concept shared among all the concrete types"

- **Ducktype**：基於模板進行文本替換的結構化規約。
  - 模板本質上就是編譯期ducktyping，不能提前獨立進行類型和語法檢查，只有實例化之後才能做類型和語法檢查，能鴨子叫，編譯通過，叫不出鴨子叫，編譯報錯。
  - C++20中模板結構化規約是concept, 此前則是SFINAE技巧，或者乾脆不成文：要麼因循舊例(iterable的T一定要有begin和end)，要麼心照不宣（自己寫的代碼，只有自己懂，無需對外公開規約）。
  - 由於規約本身並非類型，無法用普通容器存儲遵循規約的一系列具體類型，只能用類型推導的tuple-like容器。

- **Subtype**: 以基類爲規約。
  - 子類系統是面向對象語言最普及的泛型編程範式，往往基於class hierarchy +『虛表存放函數指針』+『對象內置虛指針』或『callsite胖指針』實現。
  - 子類系統固然是簡單自然的多態，但畢竟subtyping有一丟丟的性能開銷，不是零成本抽象，而且難以非侵入式地讓一個新接口適配已有代碼。

本節命名的三種範式名稱恰好都以type結尾，一方面是因爲這樣比較帥，另一方面是因爲（對C++/Rust這種抽象層次的語言來說）編程本身就是在打造一個個類型，泛型編程就是在打造一個個類型規約+遵循規約的類型。Archetype和Subtype範式中，類型規約恰好就是一種抽象類型，所以這兩種範式寫起來更簡單自然。Ducktype範式（C++模板）中，類型規約是結構化規約，可以是語言實體(concept)，也可以心照不宣，無論如何都不是類型。

| Paradigms | Archetype | Ducktype | Subtype |
| :----: | :----: | :----: | :----: |
| 具體語言實例 | Swift protocol, Carbon interface, Rust trait | C++ template(constrained or not), Rust generic | C++/Java/Python class hierarchy |
| 在哪裏寫泛型代碼？ | 泛型函數 | 函數模板 | 子類方法 |
| 承載泛型代碼的語言實體 | 普通函數，只不過是以規約類型爲參數 | 模板文本 | 普通函數，不過函數指針被語言特性暗中與callsite指針綁定了 |
| 規約 | 定義一些類的方法或屬性共性的協議類型 | 模板參數應符合的一組需求閉集，可以是成文約束Concept或SFINAE，也可以是不成文約束 | 面向對象語言繼承圖中的某個基類的虛函數 |
| 承載規約的語言實體 | 類，類型擦除之後的共性類，類型的類型，本質上仍然是類型 | 對文本替換的placeholder的結構化約束。即使約束了，我們也無法對placeholder做類型/語法檢查 | 被基類列入必要功能列表的函數 |
| 遵循規約的語言實體 | 具體的類 | 用於模板實例化的模板參數 | 子類 |
| 何時做name lookup決議？| 早綁定，允許獨立編譯| 晚綁定，實例化時，不用的話就一直不綁定| 早綁定，允許獨立編譯|
| 何時做類型檢查？| 獨立編譯時	| 實例化後	| 獨立編譯時 |
| 是否允許支持動態綁定？| 是 |否 |是 |
| 如何將已有泛型接口在新的類型上進行擴展？|	允許基於已有類型創造一個數據表示兼容的新類型，只不過規約變了，允許它實現某個新的規約，或提供與原類型的已實現的某個規約提供不同的實現。| 爲新偏特化場景寫新的函數模板即可擴展，可用Concept或SFINAE修訂重載決議規則。也可以在函數模板裏用某個約定好的函數名、成員變量名、關聯類型作爲客制點，新的類型只需實現這些客制點。| 繼承，子類數據表示可能會變 | 

## Archetype範式
Archetype在Rust中是語言的核心機制——trait，要寫好Rust的泛型代碼，就要習慣於基於archetype編程。相比與DuckType、SubType，Archetype範式天然有一些優勢。

### Generic code is just normal code
和模板相比，"Generic code is just normal code"是Archetype範式最大優勢，也是Rust語言相比C++的巨大優勢。讓非專家也能寫出高度抽象，同時還零成本，具有高度可複用性的泛型代碼。

C++基於模板文本替換的泛型代碼和普通類型的具體實現代碼，在寫法上有一定區別，編譯鏈接也有區別。
這是因爲C++中基於concept模板，其實是沒辦法提前編譯、提前語法檢查，只能在實例化之後再做語法檢查。所以是一種難度相當高的編程範式，能用模板寫C++庫的一般是業界專家，普通人很難掌握，也沒有足夠動機去掌握。

Rust基於trait的泛型代碼則和普通類型的具體實現代碼，在寫法上基本沒區別，編譯鏈接方式也一樣。
這是因爲Rust中的trait是一種archetype，type-of-type，是規定一組類型應該具備什麼接口的元類型，無論它多麼抽象多麼特殊，究其本質仍是一種類型。只要是類型，就可以單獨提前編譯，就可以被提前語法檢查。

### Adapting Erases Interoperability 
和繼承相比，"Adapting rather than extending a type"是Archetype範式的優勢之一。
Subtype範式在實踐中不能保證所有類型繼承自同一個基類，比如說對第三方的代碼沒有控制權，或者說這個類型不是個class，而是int, float這種內置類型。
Archetype範式中不僅可以爲自己的類型提供多種Archetype adaption，或使自己的代碼遵循第三方Archetype，還可以爲第三方類型提供自己的Archetype實現——前兩點還好，這最後一點是Subtype範式做不到的，只能加個醜陋的wrapper，不僅工作量特別大，而且容易出錯。

有人會問，允許修改已有的類型是不是比較危險？這是一種誤解。繼承是修改，因此是危險的。Adapt（或者說override，newtype）其實不是修改，而是新增。
繼承改變了類型的數據表示，Adapt機制則不改變類型的數據表示，只爲其新增接口——換一種說法，override Archetype for T機制實際上是爲已有類型T新建了一個遵循規約Archetype的入口。

### Archetypes in C++
有些人會argue，C++無所不能，的確C++也可以實現archetype範式，比如std::function，以及其他類型擦除。但是基於現有語法寫出來的泛型代碼和普通代碼之間還是沒那麼像。One has to drastically change the programming style in order to "go generic". 實現archetype範式在C++中相對困難。但即使如此，archetype範式的固有優勢還是讓某些標準或準標準選擇了它，比如std::function, std::any, boost::any_range。
