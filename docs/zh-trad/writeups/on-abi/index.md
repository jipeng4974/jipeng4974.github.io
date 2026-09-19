# 論ABI

> 本文總結了介於ISA和語言標準這兩個簡約協議層之間隔離了大量複雜度的抽象層次——系統語言的ABI。

---

LLMS index: [llms.txt](/llms.txt)

---

前注：ABI在本文中特指系統語言的ABI，這裏系統語言，即system programming language，指的是C，C++，Rust這樣應用於系統編程的的編譯語言。有時候我們也會討論常用庫或基礎庫的ABI，比如gcc5就把std::basic_string和std::list的實現改了，自然也就影響了跨版本的ABI兼容性，這種層面的ABI兼容性雖然也是一個工程上的疑難雜症，但不在本文的討論範圍內，畢竟源碼都不同了。本文討論ABI兼容性的是同一份源碼在不同ISA、操作系統、編譯器上對應二進制產物的兼容性。

系統語言的ABI、應用層軟件的API、微處理器架構的ISA都是描述各自抽象層次互操作能力的接口。

ABI的獨特之處在於它本身就是繁重的機制實現，而非輕量的協議聲明——實現和聲明之間有概念上的分野，這已經體現在法律上：複製ABI是赤裸裸的抄襲，而複製API或ISA則屬於fair use。你顯然可以合法地爲他人寫的API/ISA出書做注，Google實現自己的Java時合法複製了Oracle的Java API，無需license。

ABI之所以複雜，就是因爲它處於兩個抽象層次之間。ABI中的很大一部分內容既要向上對語言標準負責，向下要對ISA負責，必須遵循兩個方向的約束，將複雜度留給自身，從而成就語言標準和ISA這兩個抽象契約的簡潔、乾淨，和人類友好。

ABI中的重要組成部分之一calling convention受到ISA的約束，i386中用cdecl, stdcall, fastcall, vectorcall, thiscall, amd64中用systemv, msnative, vectorcall，arm32用aapcs，arm64用aapcs64。ABI中也有很大一部分是指令集無關的（ISA-agnostic），比如name mangling、class layout。這些機制往往足夠底層，又與其他指令集相關的部分有強耦合，加上歷史因素，往往也只有一小部分能被寫入語言標準。

如果ISA差異足夠大，維護不同的ABI無疑是必要的，強行用一套ABI兼容不僅不自然，也不高效。以早期的Windows爲例，微軟就爲早先的i386（IA-32）和後來的Intel IA-64提供了兩套ABI。再後來，AMD贏得64位戰爭，Intel也遵循了前向兼容IA-32的amd64，又名x86_64, x64，IA-64就式微了。

ABI以ISA提供的指令、寄存器、內存管理能力爲構件（building block），詳盡且精確地描述如何實現一個系統語言在特定硬件、操作系統、編譯器下的執行模型，並允許分開編譯的產物之間能互操作。例如C ABI需定義調用約定（calling convention）、基礎數據類型的表示、聚合數據類型的內存佈局；C++03 ABI需額外定義異常機制、RTTI信息存儲、虛表佈局和動態綁定機制、重載函數/運算符/模板實例化所需的名字重置（name mangling）；C++11 ABI需進一步定義lambda的實現、自動類型推導機制等新增語言機制。

無論是C，還是C++都沒有在語言層面制定官方的ABI標準，畢竟ABI標準本身也限制了實現的自由。目前最主流的Itanium C++ ABI號稱被許多操作系統採用，能適配多數微處理器架構，被大多數編譯器實現，包括gcc和clang這兩個主要玩家。但值得注意的是，被多方支持，不意味着多方支持的是完全相同的東西。同樣打着Itanium ABI名號，clang在arm32 linux上編譯的C++庫能給amd64 windows上的程序使用嗎？顯然不行，因爲最基礎的calling convention，甚至基礎數據類型表示都不同。Itanium C++ ABI即使實現大一統，積極意義也僅限於允許一些依賴CPU無關規則的危險技巧（比如修改vtable，這種操作我們現在只能稱之爲hack）在相當多的環境下通用罷了——只要C++語言標準不將ABI納入討論範圍，對ABI做出假設的技巧永遠是危險的，想要創造新的語言機制，就必須在C++標準層面推進某種ABI共識。

那麼系統語言是否應該對某種基於特定ABI實現的語言特性進行標準化呢？從C++的發展歷史來看，這種做法已經有了先例，而且是相當危險冒失的。將非零抽象開銷的dynamic exception、rtti引入C++客觀上導致了社區分裂，有相當一部分C++使用者至今依然選擇-fno-exeptions或-fno-rtti。近期的提案[Zero-Overhead Deterministic Exceptions: Catching Values](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2021/p2232r0.html)給出了一個對exception ABI進行改動的零開銷異常機制，是對歷史錯誤的亡羊補牢。系統語言的語言標準應審慎地只對ISA無關且零抽象開銷的ABI規則進行標準化，以便在此基礎上創造新的語言機制或爲應用層開發提供便利。
