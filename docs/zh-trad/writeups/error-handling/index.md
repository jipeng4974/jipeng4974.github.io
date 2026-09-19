# 錯誤處理

> 本文討論現代C++的錯誤處理問題。

---

LLMS index: [llms.txt](/llms.txt)

---

本文討論現代C++的錯誤處理問題，結論如下：
1. 宜用``Result Monad``取代C範式和C++異常，在接口層面清晰定義表明可出錯性，和錯誤清單。
2. 每個模塊應有獨立的Error類型，不宜用全局統一的Error類型或其subtype/variant，規避過於絲滑的錯誤傳播導致的舒適陷阱。
3. 上述模塊級的Error類型一般就是一個``enum class``，足夠緊湊、安全。必需在Error中留存狀態的場景再用``std::variant``+``std::visit``。
4. 明確需特殊處理的錯誤宜獨佔一個類型，不宜和常規錯誤一起並列在模塊級Error的enum class中。

# Exception Sucks in Every Possible Way
C++的異常具有多個令人絕望的特質：
- 違反C++的基本設計準則——``zero cost abstraction``。
- 鼓勵隱藏``control flow``。
- 不鼓勵在接口中給出錯誤定義。
- 對異常類型沒有任何約束。
- 不提供有界的空間時間開銷承諾。``backtrace``耗時多久完全不可預計。
- 導致編譯後的代碼膨脹。
- 不兼容``C ABI``。
- ``throw``需要動態內存分配，``catch``甚至需要``RTTI``。嵌入式環境下不會這麼奢侈。

總之C++異常機制是一種自然演化而成的歷史遺蹟，而非合乎理性的程序語言設計。以今天的標準，連寫成提案的機會都不會有就會在mailing list被C++語言律師冷嘲熱諷而消失。

哪怕現在有很多輕量化甚至零成本抽象的新型exception提案（比如P1095R0[^1]和P0709R4[^2]），一時半會兒也不會有改觀，固有的exception已經根植於歷史系統中，包含標準庫在內的龐大基礎庫都難以承受範式遷移的代價。合理對待異常的態度就是棄之不用。

# Convey Fallibility in API 
C風格error code雖然簡單，但需要使用者有強大的自控能力，對於現代編程來說還是太過危險、簡陋，語言層面本身無法區分參數中哪個是輸入，哪個是輸出，仰賴約定俗成的命名習慣或註釋，缺乏可讀性，編程時的心智負擔太重。錯誤傳播則缺乏類型安全，往往直接傳引用和指針，引發難以追蹤的內存安全、線程安全問題。不做任何錯誤傳播，就地解決錯誤的話，又會導致代碼繁瑣冗餘，總有一天會懶得做錯誤處理。

如何兼具安全的錯誤傳播、零成本抽象、接口中傳達清晰的錯誤清單？Golang也沒做到（幾乎和C一樣簡陋，給了官方的error類型，然而並非泛型或Monad，只是一個返回Error()字符串的接口）。Java有點作弊（封閉且完整的JVM+標準庫+Javadoc註釋生態圈）。Rust則輕裝簡行，爲我們展示了``Result<T,E>``的巨大成功——後續我們將其稱爲Result範式，用函數式編程語境下的``Result Monad``指代這類結構。

拋開性能、C-兼容性等問題不談，純粹從軟件系統設計的角度講，``Result Monad``也能更好地在API層面清晰無誤地將可能出現的錯誤良好定義。而exception則散落在代碼實現中，單看接口根本不知道throw了什麼，隱藏了哪些坑，如何處理這些坑——這在軟件互操作性上是災難性的。

如今Rust-like Result範式的[std::expected](https://en.cppreference.com/w/cpp/utility/expected)已經進入標準庫，自gcc12/clang16起已經可以使用了，受限於舊版編譯器的場景則可以用第三方實現。

# Keep Local Errors in Their Own Types
基於``Result<T,E>``或``expected<T,E>``進行錯誤處理時，有一個簡單方便的做法是將E設置成一個巨大的全局enum，包含所有可能的error code。這樣就可以在整個程序中自由地用``?``操作符或``and_then/or_else``傳播error，達到類似``throw``+``catch``異常的效果。

這麼做的確可以降低編程時的心智負擔，但同時也注入了濫用錯誤傳播的風險——當編碼者可以輕易甩鍋時，往往就會甩鍋。通常一個獨立且內聚的模塊會對自身內部細節有更多瞭解，有些問題還是就地處理更妥當，把失敗的細節暴露給相對外行的調用者，是在鼓勵製造不恰當的耦合。

因此每個系統模塊對外應暴露最小化的錯誤集，將內部可處理的錯誤在內部消化，並將所有``fatal error``就地解決——往往是打日誌、做些修復（有狀態系統）、退出程序。爲了避免濫用錯誤傳播，對外暴露的這個錯誤集還應該有自己的enum類型（C++語境下一般就是``enum class``，如有必要，可嘗試用``std::variant``+``std::visit``模擬Rust的``Enums`` + ``pattern matching``，見[^5]），而不宜採用全局統一的某種enum的subtype或variant——迫使直接調用者優先對錯誤進行處理而不是甩鍋給間接調用者，在甩鍋非常合理的場景下，必要的顯式類型轉換或``transform_error``/``map_error``，也讓甩鍋成爲一個有意識的system2編程決策，而非system1本能行爲。

此外，有一些特殊錯誤的處理邏輯與衆不同，不宜和其他錯誤共享一個``enum class``，而應賦予這單個error一個獨佔的類型。這種情況下，可以用嵌套的expected表示返回值類型，如``expectd<expected<T, SpecialError>, Error>``，迫使調用者對SpecialError做單獨處理。一個典型的例子是``sled``[^4]中``compare_and_swap``，返回類型特地包裹了兩層，外層是``sled::Error``，內層是特殊的``CompareAndSwapError``（CAS失敗並非異常，而是常態，對它的處理應視爲控制流，而非異常處理），在做這種設計後，``sled``用戶誤用這個函數的幾率就大大降低了，第一次用``?``操作符僅將外層的通用Error傳播了出去，不至於把必需特殊處理的CAS error也一併甩出去。

```Rust
fn compare_and_swap(
  &mut self,
  key: Key,
  old_value: Value,
  new_value: Value
) -> Result<Result<(), CompareAndSwapError>, sled::Error>

// we can actually use try `?` now
let cas_result = sled.compare_and_swap(
  "dogs",
  "pickles",
  "catfood"
)?;

if let Err(cas_error) = cas_result {
    // handle expected issue
}
```

另一個例子是需要rollback的atomic batch set操作，一旦失敗就會rollback已執行的部分set操作，避免不一致。但這個batch set操作一旦rollback失敗，則陷入無法自動恢復，必需人工干預的狀態。如果把rollback error當做KvErrorCode中的一種，只返回``std::expected<void, KvErrorCode>``，是無法迫使上游用戶區分對待rollback失敗的——調用者要麼把error統一向上傳播，要麼就簡單打個日誌，而我們希望調用者意識到數據不一致的災難已經發生，至少得打個fatal日誌。

```C++
std::expected<std::expected<void, KvErrorCode>, RollbackFatalError> BatchSet(const vec<std::pair<str, str>> &kvpairs);
```

總之，在定義類似``expected<expected<T,E1>,E2>``的``result monads``時，外層error類型``E2``永遠要比內層error類型``E1``更加fatal，更加erroneous。在避免了災難性的E2後，纔有資格判斷是否出現E1。

[^1]: [P1095R0: Zero overhead deterministic failure: A unified mechanism for C and C++](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1095r0.pdf) 
[^2]: [P0709R4: Zero-overhead deterministic exceptions: Throwing values](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0709r4.pdf) 
[^4]: [Error Handling in a Correctness-Critical Rust Project](https://sled.rs/errors) 
[^5]: [Rust enums in Modern C++ – Match Pattern](https://thatonegamedev.com/cpp/rust-enums-in-modern-cpp-match-pattern/)
