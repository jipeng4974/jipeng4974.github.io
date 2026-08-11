+++
title = "Error Handling"
date = "2024-04-09"
tags = ["SE"]
description = "This post discusses error handling in modern C++."
showFullContent = false
+++
This post discusses error handling in modern C++. The conclusions are as follows:
1. Prefer the ``Result Monad`` over C-style error codes and C++ exceptions: define fallibility and the full list of errors clearly at the interface level.
2. Each module should have its own Error type. Avoid a single global Error type, or its subtype/variant, to dodge the comfort trap created by overly smooth error propagation.
3. A module-level Error type is generally just an ``enum class`` — compact and safe enough. Reserve ``std::variant`` + ``std::visit`` for cases where the Error genuinely needs to carry state.
4. Errors that explicitly require special handling should get their own dedicated type, rather than being lumped in as a peer of the regular errors in the module-level ``enum class``.

# Exception Sucks in Every Possible Way
C++ exceptions have several hopeless traits:
- They violate one of C++'s fundamental design principles — the ``zero cost abstraction``.
- They encourage hidden ``control flow``.
- They don't encourage defining errors in interfaces.
- They impose no constraints whatsoever on exception types.
- They make no bounded space/time cost promises. How long a ``backtrace`` takes is completely unpredictable.
- They bloat the compiled code.
- They are incompatible with the ``C ABI``.
- ``throw`` requires dynamic memory allocation, and ``catch`` even requires ``RTTI``. Embedded environments can't afford such luxury.

In short, the C++ exception mechanism is a historical artifact of natural evolution, not a rational piece of programming language design. By today's standards, it wouldn't even get the chance to be written up as a proposal before being mocked into oblivion by C++ language lawyers on the mailing list.

Even though there are now many proposals for lightweight or even zero-cost exceptions (e.g., P1095R0[^1] and P0709R4[^2]), nothing will change anytime soon — the entrenched exception is rooted in historical systems, and the vast body of foundational libraries, the standard library included, can hardly bear the cost of a paradigm migration. The reasonable attitude toward exceptions is simply to stop using them.

# Convey Fallibility in API 
C-style error codes are simple, but they demand enormous self-discipline from the user. For modern programming they are still too dangerous and primitive: at the language level there is no way to tell which parameter is an input and which is an output — we rely on naming conventions or comments, which hurts readability and makes the mental burden of programming too heavy. Error propagation lacks type safety as well; references and pointers are passed around directly, inviting hard-to-trace memory safety and thread safety issues. And if you don't propagate errors at all, handling them on the spot, the code becomes tedious and redundant — sooner or later you'll be too lazy to do error handling.

How can we have safe error propagation, zero-cost abstraction, and a clearly conveyed list of errors in the interface all at once? Golang doesn't manage it (almost as primitive as C — it gives you an official error type, but it's neither generic nor a Monad, just an interface that returns an Error() string). Java kind of cheats (a closed, complete ecosystem of JVM + standard library + Javadoc comments). Rust travels light and shows us the huge success of ``Result<T,E>`` — from here on we'll call this the Result paradigm, referring to this kind of structure as the ``Result Monad`` in functional programming terms.

Setting aside performance, C-compatibility, and other concerns, purely from the standpoint of software system design, the ``Result Monad`` still does a better job of defining every possible error clearly and unambiguously at the API level. Exceptions, by contrast, are scattered throughout the implementation — looking at the interface alone, you have no idea what gets thrown, what pitfalls are hidden, or how to handle them. For software interoperability, this is catastrophic.

Today the Rust-like Result paradigm's [std::expected](https://en.cppreference.com/w/cpp/utility/expected) has entered the standard library and has been usable since gcc12/clang16; for scenarios constrained to older compilers, third-party implementations are available.

# Keep Local Errors in Their Own Types
When doing error handling based on ``Result<T,E>`` or ``expected<T,E>``, one simple and convenient approach is to make E a giant global enum containing every possible error code. That way errors can be propagated freely throughout the program with the ``?`` operator or ``and_then/or_else``, achieving an effect similar to ``throw`` + ``catch`` exceptions.

This does reduce the mental burden while programming, but it also injects the risk of abusing error propagation — when it's easy for a coder to pass the buck, they often will. Usually an independent, cohesive module knows more about its own internal details, and some problems are better handled on the spot; exposing the details of a failure to a relatively layman caller encourages inappropriate coupling.

Therefore, each system module should expose a minimal set of errors to the outside, digest internally whatever errors it can handle, and resolve every ``fatal error`` on the spot — usually by logging, doing some repair (for stateful systems), or exiting the program. To avoid abusing error propagation, this externally exposed error set should also have its own enum type (in C++ that's typically an ``enum class``; if necessary, try using ``std::variant`` + ``std::visit`` to emulate Rust's ``Enums`` + ``pattern matching``, see [^5]), rather than a subtype or variant of some globally unified enum — forcing the direct caller to handle the error first instead of buck-passing to an indirect caller. In scenarios where buck-passing is perfectly reasonable, the necessary explicit type conversion or ``transform_error``/``map_error`` also turns it into a deliberate system2 programming decision rather than a system1 reflex.

Moreover, some special errors have handling logic that differs from the rest; they should not share an ``enum class`` with the other errors, and this single error should be given its own dedicated type instead. In this case you can use a nested expected for the return type, e.g. ``expectd<expected<T, SpecialError>, Error>``, forcing the caller to handle SpecialError separately. A classic example is ``compare_and_swap`` in ``sled``[^4]: its return type is deliberately wrapped in two layers — the outer one is ``sled::Error`` and the inner one is the special ``CompareAndSwapError`` (a CAS failure is not an exception but the norm, and handling it should be treated as control flow, not exception handling). With this design, ``sled`` users are far less likely to misuse the function: the first use of the ``?`` operator only propagates the outer, generic Error, so the CAS error that must be specially handled doesn't get tossed out along with it.

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

Another example is an atomic batch set operation that needs rollback: once it fails, the set operations already executed are rolled back to avoid inconsistency. But if the batch set operation's rollback itself fails, it falls into a state that cannot be recovered automatically and requires manual intervention. If you treat the rollback error as just one of the KvErrorCode values and only return ``std::expected<void, KvErrorCode>``, there is no way to force upstream users to treat rollback failure differently — the caller either propagates all errors upward uniformly or simply writes a log line, whereas we want the caller to realize that the catastrophe of data inconsistency has already happened and at least write a fatal log.

```C++
std::expected<std::expected<void, KvErrorCode>, RollbackFatalError> BatchSet(const vec<std::pair<str, str>> &kvpairs);
```

In short, when defining ``result monads`` like ``expected<expected<T,E1>,E2>``, the outer error type ``E2`` should always be more fatal, more erroneous than the inner error type ``E1``. Only after the catastrophic E2 has been ruled out are you in a position to judge whether E1 occurred.

[^1]: [P1095R0: Zero overhead deterministic failure: A unified mechanism for C and C++](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1095r0.pdf) 
[^2]: [P0709R4: Zero-overhead deterministic exceptions: Throwing values](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/p0709r4.pdf) 
[^4]: [Error Handling in a Correctness-Critical Rust Project](https://sled.rs/errors) 
[^5]: [Rust enums in Modern C++ – Match Pattern](https://thatonegamedev.com/cpp/rust-enums-in-modern-cpp-match-pattern/)
