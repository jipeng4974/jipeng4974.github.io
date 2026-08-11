+++
title = "String Lookups Reduce to Parsing"
date = "2023-05-14"
tags = ["Sys"]
description = "String lookup and string parsing both essentially extract state from a character stream using the most compact structure and the most efficient algorithm possible. So the NFA-to-DFA algorithm from the Dragon Book can be put to good use."
showFullContent = false
+++

The title is the conclusion: the string lookup problem can be reduced to a parsing problem.

This conclusion comes from a recent interesting observation: writing a high-performance ASCII protocol parser with ragel is essentially about converting an [NFA](https://en.wikipedia.org/wiki/Nondeterministic_finite_automata) into a [DFA](https://en.wikipedia.org/wiki/Deterministic_finite_automaton) for performance, and this is isomorphic to the approach in [toplingdb](https://github.com/topling/toplingdb) of merging the tries (essentially DFAs) corresponding to the individual SSTs at the same level into a single DFA (many DFAs -> NFA -> 1 DFA).

This isomorphism quietly implies a reduction: string lookup and string parsing both essentially extract state from a character stream using the most compact structure and the most efficient algorithm possible, so lookups can be regarded as a special kind of parsing (with quite regular patterns). LSM key lookup is an even more special case — a massive number of indexes whose key ranges have no overlap — where there are plenty of DFAs that can be merged easily. So the many NFA-to-DFA algorithms from the Dragon Book can be put to good use.

A KV store's in-memory key index can be a red-black tree, a skiplist, a hashmap, arguably a patricia trie (a variant of the [radix tree](https://en.wikipedia.org/wiki/Radix_tree)), or the NestLoudsTrie in toplingdb. The same kind of structure can also be found in routing table implementations. String indexing, in the end, comes down to a string lookup structure. Just as a routing table implementation can merge all its routes into a single DFA (many routes contain regexes), a KV database can also merge multiple indexes of the Trie — a special kind of DFA (a Trie's state transition graph is a tree, and a tree is an undirected graph with the extra constraint that any two nodes are connected by only one edge) — where the key range corresponding to each index doesn't even overlap (an LSM property). As a result, merging is very fast, and the merged DFA is simple and compact to represent. For details, see [The Application of Automaton Algorithms in Database Indexes](https://zhuanlan.zhihu.com/p/628057993). I followed up in the comments of the author's article about the trigger conditions and the overhead of DFA merging, and the author replied that merging is triggered at compaction/flush time, accounts for a very small share of the whole LSM update process, and involves no multithreading, so thread safety is not a concern.
