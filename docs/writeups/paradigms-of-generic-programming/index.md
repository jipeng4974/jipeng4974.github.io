# Paradigms of Generic Programming: Archetype, Ducktype, Subtype

> This post summarizes the three paradigms of generic programming: Archetype, Ducktype, and Subtype. All three names end in "type" — partly because it looks cool, with a sense of regularity and the architectural beauty of logic, and partly because systems language programming is itself about building types, while generic programming is about building type specifications plus the types that conform to them.

---

LLMS index: [llms.txt](/llms.txt)

---

## Generic Is Not Synonymous with Template
What is generic programming? When people hear "generic programming", many immediately think of templates, but template programming is only one paradigm of generic programming. Alexander Stepanov, who coined the term "generic programming", repeatedly emphasized:
generic programming is not about how to use templates. The first version of the STL, despite having "template" in its name, was actually implemented on top of Scheme.

## To Be Generic, You Need Specifications
Defining specifications and conforming to them is the foundation of generic programming.
Any kind of generic programming requires some specification; only with a specification can one abstraction be adapted to multiple concrete entities. Without a specification, the people writing concrete implementations would have no idea what rules to follow, what interfaces to implement, or what conditions to satisfy — and generic programming would naturally be out of the question.

Specifications: in JavaScript, prototypes; in Swift, protocols; in Rust, traits; in C++ template programming, concepts or SFINAE.

Conforming to specifications: in JavaScript, linking a concrete object to a prototype object at runtime; in Swift, making a concrete type conform to certain protocols with an inheritance-like colon; in Rust, explicitly providing an implementation of a trait for a type with the `impl SomeTrait for SomeStruct` syntax; in C++, template arguments need no special syntax to declare that they satisfy the template parameters, but they must tacitly comply with the requirements of the template code.

## Three Specifications, Three Paradigms
Depending on the kind of specification, we can name three paradigms of generic programming: Archetype, Ducktype, and Subtype.
- **Archetype**: takes a type-of-type — in other words, a protocol or interface — as the specification.
  - [Rust trait](https://doc.rust-lang.org/book/ch10-02-traits.html): "defines shared behavior"
  - [Carbon interface](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/generics/terminology.md#interface): "defines an API that a given type can implement"
  - [Swift protocol](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/protocols/): "defines a blueprint of methods, properties, and other requirements"
  - [C++ type erasure idiom](https://davekilian.com/cpp-type-erasure.html): "captures the concept shared among all the concrete types"

- **Ducktype**: a structural specification based on textual substitution in templates.
  - Templates are essentially compile-time duck typing: type and syntax checking cannot be done independently in advance; they can only be performed after instantiation. If it quacks like a duck, it compiles; if it can't quack, you get a compile error.
  - In C++20, the structural specification for templates is the concept; before that it was SFINAE tricks, or simply unwritten conventions: either following established precedent (an iterable `T` must have `begin` and `end`), or tacit understanding (code you wrote yourself that only you understand, with no need to publish a specification).
  - Since the specification itself is not a type, you cannot store a series of concrete types conforming to it in an ordinary container; you can only use type-deduced tuple-like containers.

- **Subtype**: takes a base class as the specification.
  - Subclassing is the most widespread generic programming paradigm in object-oriented languages, usually implemented as class hierarchy + "virtual tables storing function pointers" + "objects with built-in vtable pointers" or "fat pointers at the call site".
  - Subclassing is admittedly simple and natural polymorphism, but subtyping does carry a slight performance overhead — it is not a zero-cost abstraction — and it is hard to adapt a new interface to existing code in a non-intrusive way.

The three paradigm names chosen in this section all happen to end in "type" — partly because it looks cool, and partly because (for languages at the abstraction level of C++/Rust) programming is itself about building types, and generic programming is about building type specifications plus the types that conform to them. In the Archetype and Subtype paradigms, the type specification happens to itself be an abstract type, so these two paradigms are simpler and more natural to write. In the Ducktype paradigm (C++ templates), the type specification is a structural specification: it can be a language entity (concepts), or it can remain tacit — but either way it is not a type.

| Paradigms | Archetype | Ducktype | Subtype |
| :----: | :----: | :----: | :----: |
| Concrete language instances | Swift protocol, Carbon interface, Rust trait | C++ template (constrained or not), Rust generics | C++/Java/Python class hierarchy |
| Where do you write generic code? | Generic functions | Function templates | Subclass methods |
| Language entity carrying the generic code | Ordinary functions, just parameterized by the specification type | Template text | Ordinary functions, except the language feature secretly binds the function pointer to the call-site pointer |
| Specification | A protocol type defining commonalities of methods or properties across classes | A closed set of requirements that template arguments must satisfy — either written constraints (concepts or SFINAE) or unwritten ones | Virtual functions of some base class in the inheritance graph of an OO language |
| Language entity carrying the specification | A class — the common class after type erasure, a type-of-type, essentially still a type | Structural constraints on the placeholders of textual substitution; even when constrained, we still cannot type/syntax-check the placeholders | Functions listed by the base class as required functionality |
| Language entity conforming to the specification | Concrete classes | Template arguments used for template instantiation | Subclasses |
| When is name lookup resolved? | Early binding; allows separate compilation | Late binding, at instantiation time; never bound if never used | Early binding; allows separate compilation |
| When is type checking done? | At separate compilation | After instantiation | At separate compilation |
| Is dynamic binding supported? | Yes | No | Yes |
| How do you extend an existing generic interface to new types? | You may create a new type with a data representation compatible with an existing type — only the specification changes; the new type may implement a new specification, or provide a different implementation of a specification the original type already implements. | Just write new function templates for the new partial specialization scenario; concepts or SFINAE can be used to adjust the overload resolution rules. You can also use agreed-upon function names, member variable names, or associated types inside function templates as customization points — new types only need to implement these customization points. | Inheritance; the subclass's data representation may change |

## The Archetype Paradigm
In Rust, the archetype is the language's core mechanism — the trait. To write good generic Rust code, you need to get used to archetype-based programming. Compared with Ducktype and Subtype, the Archetype paradigm has some inherent advantages.

### Generic code is just normal code
Compared with templates, "generic code is just normal code" is the greatest advantage of the Archetype paradigm, and a huge advantage of Rust over C++. It lets non-experts write highly abstract, zero-cost, highly reusable generic code.

In C++, generic code based on template textual substitution differs somewhat from concrete implementation code for ordinary types, both in how it is written and in how it is compiled and linked.
This is because in C++, even concept-constrained templates cannot be compiled or syntax-checked in advance — syntax checking can only happen after instantiation. So it is a rather difficult programming paradigm: those who can write C++ libraries with templates are usually industry experts; ordinary people find it hard to master and lack sufficient motivation to do so.

Rust's trait-based generic code, on the other hand, is basically indistinguishable from concrete implementation code for ordinary types, both in writing style and in how it is compiled and linked.
This is because a Rust trait is an archetype, a type-of-type: a meta-type that specifies what interfaces a group of types should provide. No matter how abstract or special it is, it is essentially still a type. And as long as it is a type, it can be compiled separately in advance and syntax-checked in advance.

### Adapting Erases Interoperability 
Compared with inheritance, "adapting rather than extending a type" is one of the Archetype paradigm's advantages.
In practice, the Subtype paradigm cannot guarantee that all types inherit from the same base class — for instance, you may have no control over third-party code, or the type may not be a class at all but a built-in type like `int` or `float`.
With the Archetype paradigm, you can not only provide multiple Archetype adaptions for your own types, or make your own code conform to third-party Archetypes — you can also provide your own Archetype implementations for third-party types. The first two are fine; this last one is something the Subtype paradigm cannot do — you can only add an ugly wrapper, which is a lot of work and error-prone.

Some may ask: isn't allowing modification of existing types dangerous? That is a misunderstanding. Inheritance is modification, and therefore dangerous. Adapting (or overriding, newtype) is not modification but addition.
Inheritance changes a type's data representation; the Adapt mechanism does not change the type's data representation — it only adds new interfaces to it. Put another way, the "override Archetype for T" mechanism actually creates a new entry point through which the existing type T conforms to the Archetype specification.

### Archetypes in C++
Some people argue that C++ can do anything, and indeed C++ can also realize the archetype paradigm — e.g. `std::function` and other type erasure facilities. But generic code written with the existing syntax still doesn't look that much like ordinary code. One has to drastically change the programming style in order to "go generic". Implementing the archetype paradigm is relatively difficult in C++. Even so, the inherent advantages of the archetype paradigm led certain standard or quasi-standard facilities to choose it, such as `std::function`, `std::any`, and `boost::any_range`.
