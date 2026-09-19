# C++系統工程範式

> 本文總結當下我認爲比較好的C++系統工程範式。

---

LLMS index: [llms.txt](/llms.txt)

---

## 基於真實需求，創造新穎的計算形態。
- 若一個系統不存在新穎性，則不必爲審美或宗教原因重複造輪。
- 應認識到，絕大多數系統在商業上幾乎沒有價值，本就不應該存在。自然不應該向本就不該存在的系統投入稀缺的工程人力。

## 用最小技術集構建當前問題的直接解。
- 如果已有技術修修補補後只能勉強解決問題，存在不可克服的缺陷，則考慮研發新系統作爲直接解。
- 不爲未來的不確定需求破壞當前方案的簡潔性。
- 管理複雜度，提升可理解性和可維護性。
  - 核心代碼fit in人腦短期記憶的解決方案是技術資產，反之則是技術債。
  - 若核心代碼複雜度高到無法fit in單人短期記憶，則將其合理切分成多個模塊，交由多人維護。
  - 避免引入過多的第三方庫，規避C++ dependency hell。

## 避免解釋型註釋，讓代碼自解釋。
- 如果某一行代碼做了什麼需要註釋，那麼說明它還可以進行更好的重構。
  - 這種註釋就像todo標誌，表明有空就應該把它重構一下，提升代碼質量的同時順便去除註釋。
  - 剔除這類註釋，可以避免日後迭代中，註釋逐漸和代碼對不上，對讀者產生災難性的誤導。
- 儘量闡釋爲何這麼做(WHY)，而不註解做了什麼(WHAT)。
  - 比較複雜的場景，可以用大段文字整體介紹設計思路。
  - 這段文字可以放在代碼最前面醒目位置，方便隨着版本更新也進行相應更新。不必寫額外的文檔，或Readme，因爲外置的文檔和代碼之間的映射也是脆弱、難以長期維護的。
- 兼顧人體工學與性能工程，在不同場合做不同的取捨：
  - nonexpert-transparent：讓大多數代碼邏輯足夠簡單易懂、簡潔直接，以至於任何背景的研發都能不借助文檔、註釋輕鬆讀懂代碼。
  - expert-friendly：與此同時，代碼庫中性能攸關部分，必須能允許專家施加任意強度的優化。任何人體工學驅動的抽象都不能影響到性能工程的自由度。

## 使每個編譯單元內自帶文檔、單測和構建指令
- 編譯單元(.cc和它包含的頭文件)承載了某個功能的實現，但對這個功能的測試、文檔描述、構建腳本卻往往放在其他文件裏，有時會在很奇怪的位置，甚至混雜在其他複雜文件中，對於新接手項目的人來說了解這個編譯單元的全貌就變得很麻煩。
- 所以不妨直接把單測代碼寫在每個.cc文件最下面，把單測的構建信息以註解形式寫在.cc文件最上方，把文檔以大段註釋的方式寫在.cc/.h的醒目位置。
- 如無特殊需求，構建工具不妨就用[emake](https://github.com/skywind3000/emake)。

## 非性能攸關場景，儘可能拆分出更多函數，讓每個函數只做一件事。
- 如果一個複雜函數能夠拆解成多個函數，則將其拆解。
- 如果多個函數共享一些變量，則重構成類。

## 儘可能避免使用exception，而是使用result monad。
- 避免異常引入不必要的性能開銷。
- 唯有禁用異常，才能convey fallibility through APIs。
- std::expected<T,E>或best::result<T,E>都是不錯的選擇。
  - 問題是ctor沒有返回值，這其實也是最初C++需要設計exception的原因。作爲workaround，可以選擇提供一個static make/create函數，返回expected<T,E>。
  - 順便避免copy ctor。大多數類不需要拷貝構造，少數需要的情況應顯式實現一個copy/clone函數。
  - C++沒有Rust的?語法糖，但可以自己實現一個try宏[^5]。
- 非對外接口，非性能攸關，錯誤處理並不重要，且只需向上傳播的少數場景，宜用exception——且最好用gcc14.2+[^3]（對exception性能大有提升）。
  - 這樣的場景並非完全不存在，典型的例子是json parsing，你並不需要對每個位置、各種情形的json錯誤做出不同的應對，大多數時候只需要打印出來哪一行json格式不對就行了。
  - 此外，各種一次性腳本、臨時任務、或簡單應用等速朽場景，甚至連exception都不一定要用，自然也不需要general-purpose library或data-center application級別的error handling和接口設計，能用就行。

## 設計之初就將系統中的錯誤分類，明確責任邊界
- 錯誤可分爲：用戶錯誤（invalid_input）、可容忍系統錯誤（glitch）、不可容忍系統錯誤（fatal）、編程錯誤（bug）。
- 用戶輸入錯誤輸入是常態，應將其視爲正常路徑處理，不應使用exception，不用做任何補救和恢復，儘快返回錯誤消息讓用戶知悉即可。
- 系統錯誤是某個下游組件或某個系統調用失敗導致的錯誤，這種錯誤應被視爲系統故障，其中不可容忍的嚴重故障應與普通故障區分開——以便給它們單獨的日誌等級，引入某種告警和人工介入機制。
- 編程錯誤是理想代碼中本不應該出現的斷言失敗，有些是precondition檢查失敗，有些是postcondition檢查失敗，這些都可以歸類爲bugs，發現後需進行修復。

## 慎重對待需要就地處理的局部錯誤。
- 錯誤是分層的，處理錯誤的context也是分層的，有些錯誤只能在它的上一層級得到妥善處理，一旦向上層傳播，就會丟失正確處理它的context，因此有必要從接口設計層面慎重對待這類錯誤。
- 如果整個項目的std::expected<T,E>中的E都是一個全局錯誤類型，編程時可以非常輕鬆地進行error propagation。但這也意味着有些不該拋給上層的錯誤也容易被調用者拋上去。
- 應賦予需要就地處理的局部錯誤一個獨佔的類型[^1]，並用嵌套的expected表示返回值類型，如``expectd<expected<T, SpecialError>, Error>``，迫使調用者對SpecialError做單獨處理。

## 用FTADLE實現多態。
- 我們當然不會用笨重的繼承+動態綁定（subtype），也少用醜陋的模板+concept（ducktype）。
- 相比而言，FTADLE是更加簡潔、靈活、美觀、bug-free且易維護的定製化方案，巧妙利用了C++的一個冷門語言特性（[ADL](https://en.cppreference.com/w/cpp/language/adl)），實現了一種優雅的archetype多態。[^2]

## 讓自定義類型儘可能平凡（trivial）
- 能安全的放進各種容器，即copy assignable + copy constructible。
- 按bits複製（比如std::memcpy）可正確完成對象的複製，即trivially copyable。
- 少數確實不宜設計爲copyable的類型，保證trivally move assignable/constructible。
- 可無異常默認構造，即nothrow default constructible。
## 用強類型表達約束
- 強類型作爲參數，可以消除一些函數的隱式的前置條件（implicit preconditions）。
  - 比如std::string的back()函數返回char&，隱含了該string非空的前置條件。[^4]
  - 很多參數應該屬於它的類型，但並不是所有這個類型的值都是合法參數，比如path用std::string來存很合理，但不是所有string都是合法path。
  - 用non_empty_string作爲string類型，就可以消除非空這一前置條件。
  - 類似地，可以用更特殊的類型將表達各種約束。
- 在頭文件中定義好約束更強更安全的基礎類型。
  - 比如禁止和有符號數加減轉換的8bits無符號數：``using u8 = type_safe::integer<uint8_t>;``
  - 比如禁止從0/1轉換的布爾類型：``using boolean = type_safe::boolean;``
  - 比如禁止用operator==()的浮點數類型：``using f32 = type_safe::floating_point<float>;``


[^1]: [Error Handling](https://jipeng4974.github.io/writeups/error-handling)
[^2]: [Paradigms of Generic Programming: Archetype, Ducktype, Subtype](https://jipeng4974.github.io/writeups/paradigms-of-generic-programming/)
[^3]: [C++ exception performance three years later](https://databasearchitects.blogspot.com/2024/12/c-exception-performance-three-years.html)
[^4]: [Prevent precondition errors with the C++ type system](https://www.foonathan.net/2016/09/error-handling-types/)
[^5]: 可以考慮基於[StatementExprs](https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html)實現try宏，模擬Rust的?操作符，clang也有類似機制。
