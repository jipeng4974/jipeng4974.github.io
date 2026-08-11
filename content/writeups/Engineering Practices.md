+++
title = "Idiomatic Practices in C++ Systems Engineering"
date = "2024-12-03"
tags = ["sys", "se"]
description = "A summary of the C++ systems engineering paradigms I currently consider sound."
showFullContent = false
+++

## Build novel computational forms based on real needs.
- If a system has no novelty, don't reinvent the wheel for aesthetic or religious reasons.
- Recognize that the vast majority of systems have almost no commercial value and should never have existed in the first place. Naturally, scarce engineering manpower should not be poured into systems that shouldn't exist.

## Build a direct solution to the problem at hand with a minimal set of technologies.
- If existing technologies can only barely solve the problem after endless patching, with flaws that cannot be overcome, consider developing a new system as the direct solution.
- Don't break the simplicity of the current solution for uncertain future requirements.
- Manage complexity; improve comprehensibility and maintainability.
  - A solution whose core code fits in a person's short-term memory is a technical asset; otherwise it is technical debt.
  - If the core code is too complex to fit in a single person's short-term memory, split it sensibly into multiple modules and assign them to multiple maintainers.
  - Avoid pulling in too many third-party libraries; steer clear of C++ dependency hell.

## Avoid explanatory comments; make the code self-explanatory.
- If a line of code needs a comment to explain what it does, that means it can be refactored better.
  - Such comments are like todo markers, indicating that the code should be refactored when there's time — improving code quality while removing the comment along the way.
  - Eliminating this kind of comment avoids the catastrophic misleading of readers when comments gradually drift out of sync with the code during later iterations.
- Explain why it's done this way (WHY) rather than annotating what is done (WHAT).
  - For more complex scenarios, a long block of text can introduce the overall design rationale.
  - This text can sit in a prominent place at the top of the code, so it gets updated along with version changes. No extra documents or READMEs are needed, because the mapping between external documents and the code is also fragile and hard to maintain over the long term.
- Balance ergonomics against performance engineering, making different trade-offs in different contexts:
  - nonexpert-transparent: make most of the code logic simple, clear, concise, and straightforward enough that developers of any background can easily read it without documents or comments.
  - expert-friendly: at the same time, the performance-critical parts of the codebase must allow experts to apply optimizations of arbitrary intensity. No ergonomics-driven abstraction may constrain the freedom of performance engineering.

## Ship documentation, unit tests, and build instructions inside every compilation unit
- A compilation unit (the .cc file and the headers it includes) carries the implementation of some functionality, but its tests, documentation, and build scripts are often placed in other files, sometimes in very strange locations, or even mixed into other complex files — making it a hassle for someone new to the project to grasp the full picture of that compilation unit.
- So you might as well write the unit tests at the bottom of each .cc file, put the test build information as an annotation at the top of the .cc file, and write the documentation as a long comment block in a prominent spot in the .cc/.h file.
- Unless there are special requirements, [emake](https://github.com/skywind3000/emake) works fine as the build tool.

## In non-performance-critical code, split out as many functions as possible so that each function does only one thing.
- If a complex function can be broken down into multiple functions, break it down.
- If multiple functions share some variables, refactor them into a class.

## Avoid exceptions whenever possible; use the result monad instead.
- Avoid the unnecessary performance overhead introduced by exceptions.
- Only by disabling exceptions can you convey fallibility through APIs.
- std::expected<T,E> or best::result<T,E> are both good choices.
  - The catch is that ctors return nothing — which is precisely why C++ needed exceptions in the first place. As a workaround, provide a static make/create function that returns expected<T,E>.
  - This also sidesteps the copy ctor. Most classes don't need a copy constructor; for the few cases that do, implement an explicit copy/clone function instead.
  - C++ lacks Rust's ? syntactic sugar, but you can implement your own try macro[^5].
- For the few cases that are not external interfaces, not performance-critical, where error handling doesn't matter much and errors only need to propagate upward, exceptions are appropriate — and preferably with gcc14.2+[^3] (which greatly improves exception performance).
  - Such cases are not entirely nonexistent; a typical example is JSON parsing. You don't need to handle every JSON error at every position differently — most of the time you just need to print which line has malformed JSON.
  - Besides, for throwaway scenarios like one-off scripts, ad-hoc tasks, or simple applications, you may not even need exceptions, let alone general-purpose-library or data-center-application-grade error handling and interface design. Good enough is good enough.

## Classify the errors in the system from day one, and define responsibility boundaries clearly
- Errors can be classified into: user errors (invalid_input), tolerable system errors (glitch), intolerable system errors (fatal), and programming errors (bug).
- Users entering invalid input is the norm; treat it as the normal path. Don't use exceptions, don't attempt any remediation or recovery — just return an error message as quickly as possible to inform the user.
- A system error is caused by a failing downstream component or a failing system call. Such errors should be regarded as system failures, and intolerable severe failures should be distinguished from ordinary ones — so they can get their own log level and trigger some alerting and human-intervention mechanism.
- A programming error is an assertion failure that should never occur in ideal code. Some are failed precondition checks, some are failed postcondition checks; all of these can be classified as bugs and must be fixed once discovered.

## Treat local errors that must be handled in place with care.
- Errors are layered, and so is the context for handling them. Some errors can only be properly handled at the level right above them; once propagated further up, the context needed to handle them correctly is lost. So this class of errors deserves careful treatment at the interface-design level.
- If the E in std::expected<T,E> is a single global error type across the whole project, error propagation becomes very easy while programming. But that also means errors that shouldn't be thrown upward can easily be thrown up by callers.
- Give local errors that must be handled in place a dedicated type[^1], and express the return type with a nested expected, e.g. ``expectd<expected<T, SpecialError>, Error>``, forcing the caller to handle SpecialError separately.

## Implement polymorphism with FTADLE.
- We certainly won't use the clunky inheritance + dynamic binding (subtype), and we rarely use the ugly template + concept (ducktype).
- By comparison, FTADLE is a more concise, flexible, elegant, bug-free, and maintainable bespoke approach. It cleverly exploits an obscure C++ language feature ([ADL](https://en.cppreference.com/w/cpp/language/adl)) to achieve a graceful archetype polymorphism.[^2]

## Make user-defined types as trivial as possible
- Safe to put into all kinds of containers, i.e. copy assignable + copy constructible.
- Bitwise copying (e.g. std::memcpy) correctly copies the object, i.e. trivially copyable.
- For the few types that genuinely shouldn't be copyable, guarantee trivially move assignable/constructible.
- Default-constructible without throwing, i.e. nothrow default constructible.
## Express constraints with strong types
- Strong types as parameters can eliminate a function's implicit preconditions.
  - For example, std::string's back() returns char&, implicitly assuming the precondition that the string is non-empty.[^4]
  - Many parameters should belong to their type, but not every value of that type is a valid argument — for instance, storing a path in a std::string is reasonable, but not every string is a valid path.
  - Using non_empty_string as the string type eliminates the non-emptiness precondition.
  - Similarly, more specialized types can be used to express all sorts of constraints.
- Define more strongly constrained, safer fundamental types in header files.
  - For example, an 8-bit unsigned integer that forbids arithmetic conversions with signed numbers: ``using u8 = type_safe::integer<uint8_t>;``
  - For example, a boolean type that forbids conversion from 0/1: ``using boolean = type_safe::boolean;``
  - For example, a floating-point type that forbids operator==(): ``using f32 = type_safe::floating_point<float>;``


[^1]: [Error Handling](https://jipeng4974.github.io/writeups/error-handling)
[^2]: [Paradigms of Generic Programming: Archetype, Ducktype, Subtype](https://jipeng4974.github.io/writeups/paradigms-of-generic-programming/)
[^3]: [C++ exception performance three years later](https://databasearchitects.blogspot.com/2024/12/c-exception-performance-three-years.html)
[^4]: [Prevent precondition errors with the C++ type system](https://www.foonathan.net/2016/09/error-handling-types/)
[^5]: One option is to implement the try macro based on [StatementExprs](https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html), emulating Rust's ? operator; clang has a similar mechanism.
