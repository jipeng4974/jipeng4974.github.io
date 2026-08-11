# Linkers & Loaders

> Linkers & Loaders fills a niche body of knowledge — linking and loading.

---

LLMS index: [llms.txt](/llms.txt)

---

The basic job of a linker or loader is binding — binding abstract names to more concrete ones, such as binding the `getline` function to "byte 612 of .text".

## A History of Address Binding
In the era of punched card/paper tape computers, programmers hand-assembled symbolic programs into machine code and fed it into the machine. If the code used names (symbolic addresses), the programmer had to translate them into addresses by hand. As a result, adding or removing any single instruction in the code could affect every address in the machine code.

This is the bitter fruit of binding names to addresses too early. Assemblers solved this problem by allowing programmers to write programs using symbolic names.

The punched card era already had the concepts of subroutines and libraries. Subroutines were stored sorted into card decks, and when the main program needed a subroutine, the corresponding deck had to be loaded and rearranged. This process was essentially manual `library search` and `relocation`.

Before operating systems existed, every program assumed it had the entire machine's memory to itself, so it could naturally use fixed memory addresses — after all, every address on the machine was available. Once operating systems appeared, programs had to share memory with the OS and even with other programs, and the actual addresses could only be known after the OS loaded the program. This pushed address binding back from link time to load time, and the relocating loader split off from the linker. The linker performs part of the address binding, reallocating relative addresses within each program; the loader handles relocation, performing the final address assignment.

Early memory was extremely scarce, and programs soon outgrew the memory limit, so linkers provided an overlay mechanism that allowed different parts of a program to share the same block of memory. This mechanism did not disappear until virtual memory arrived in the 90s. Hardware relocation and virtual memory made linkers and loaders simpler.

When a computer runs multiple copies of the same program, most of the program's content can actually be shared, so the segmentation mechanism was introduced to separate read-only code segments from writable segments — only one copy of the read-only code segment needs to be loaded on a machine. The linker therefore has to allocate addresses for each segment separately.

Even when a computer runs many different programs, those programs often share large amounts of code, which led to shared libraries. Static shared libraries were not flexible enough: any code change in the library required relinking. So dynamic shared libraries appeared, in which symbols and segments are not bound to actual addresses — the binding is deferred until the program runs, and can even be delayed further, until the first call.

## Linking vs Loading
The linker handles symbol resolution; the loader handles program loading. Both can do relocation, and there are also all-in-one `linking loaders`.

- Symbol resolution: symbols are the medium through which a program calls subroutines. The linker resolves function names like sqrt to locations in a library and patches the caller's code so the call instruction points to that location.
- Program loading: loading copies the program into memory, and along the way also sets memory protection bits, arranges virtual memory mappings, and so on.
- Relocation: compilers and assemblers generate program addresses starting from zero for each compilation unit (file). When the linker combines multiple subroutines into a complete program, it usually performs one round of relocation. The complete program's addresses still start from zero, so after the loader loads the program into memory, it performs relocation once more.

## 2-pass Linking
Linking, like compiling and assembling, is a 2-pass process. The linker takes object files, static libraries, dynamic libraries, and command-line arguments as input, and produces an executable file — plus, when debugging is enabled, debugger symbol files or a load map. Object files, static libraries, and dynamic libraries are all segmented (code/data) and each contains at least one symbol table, exporting and importing some symbols.

In the 1st pass, the linker scans all input files, obtains the size of each segment, and collects all symbol definitions and references, thereby creating a unified segment table and a unified symbol table. It then assigns a location to every symbol and determines the sizes and positions of the segments in the output address space.

In the 2nd pass, the linker reads the previously generated object files, replaces all symbol references with numeric addresses, adjusts all memory addresses in code and data to the relocated segment addresses, and finally adds the header, relocation sections, and symbol table to the updated object file.

If the program uses dynamic linking, the symbol table contains the information the `runtime linker` needs to resolve dynamic symbols. The linker usually also generates some glue code that provides calling stubs for invoking dynamically linked libraries.

Whether or not the program uses dynamic linking, the symbol table always provides some information for relinking and debugging — many object formats are relinkable, meaning the generated object files are allowed to serve as input to later links.

## Object Files
The binary code files that compilers and assemblers generate from source code are object files. They contain a header, object code, a relocation list (positions that need to be relocated at link time), a global symbol table, debugging information, and more.

As raw material, object files ultimately feed into three kinds of end products: linkables, executables, and loadables.

- Linkables contain rich symbol information and relocation information, and their object code is organized into fine-grained logical sections, making it convenient for the linker to post-process them with symbol resolution and relocation.
- Executables contain page-aligned object code (allowing it to be mapped into virtual memory), need not provide any symbol information beyond what dynamic linking requires, and need to provide little or no relocation information. Their object code is organized into coarser-grained segments, or into segments reflecting the specific hardware execution environment, often split into read-only and read-write pages.
- Loadables may only need to contain object code, or may need to provide full symbol and relocation information, depending on the implementation of the system runtime.

The typical object file format, Unix a.out, contains a header, a text section, a data section, and other sections.
Its header (taking BSD as an example) contains the text segment size, initialized data size, uninitialized data size (the BSS segment), symbol table size, entry point (starting address), text relocation size, and data relocation size.

When loading an a.out, the operating system first reads the header to get the size of each segment, then checks whether a shared code segment already exists and creates one if not. It maps the text segment into the memory space, creates a private data segment large enough, initializes the bss segment to zero, creates and maps a stack segment (often separate from the data segment, since heap and stack typically grow at different rates), pushes the program's initial arguments onto the stack, and finally sets up the registers and jumps to the program's starting address.

To reduce unnecessary paging and let object files map directly onto 4K pages, later UNIX systems introduced pageable formats that expand the header to 4K and round the text segment boundary up to the next 4K. The downside is that they are not compact and waste disk space. Later still came compact pageable formats that simply treat the header as part of the text segment (QMAGIC and ELF).

a.out does not support relocation, nor does it support the special handling of C++ initializer/finalizer code, so it was replaced by ELF (Executable and Linking Format), which supports cross-compilation, dynamic linking, and more.

ELF adopted DWARF as its debugging format and provides three file types: relocatable, executable, and shared object.
- Relocatables can be created by compilers and assemblers, but must be further linked before they can run.
- Executables have completed relocation and static symbol resolution and can be mapped directly into memory.
- Shared objects are dynamically linked libraries, containing both the symbol information needed at link time and code executable at runtime.

ELF is designed with a dual nature:
- From the loading perspective, it is loadable segments about to be placed into memory: to the loader, an ELF is a set of segments described by the program header, and there is no need to care about sections. There are only a handful of loadable segments. A typical Linux ELF linked by BFD-ld or Gold is generally divided into 2 loadable segments: RE (read-execute, containing .text, .rodata, etc.) and RW (read-write, containing .data, .bss, etc.). This reduces the number of kernel mmaps to 2, but putting read-only data into a read-execute segment sacrifices some security. Newer Linux systems, out of security considerations, split it into 3 segments: R, RE, and RW, placing the ELF header and .rodata in R. Even newer BFD-ld, while putting the ELF header and .rodata in the R segment, forgot to merge the two Rs, resulting in 4 loadable segments.[^1]
- From the linking perspective, it is linkable sections on disk: the section mechanism allows the linker to further process the ELF. A single segment consists of several sections. For example, one loadable read-only segment can contain three sections: executable code, read-only data, and dynamic linking symbols. Relocatables have a section table. Executables have an ELF header table. Shared objects have both. A typical ELF relocatable program contains more than a dozen sections, such as .text, .data, .rodata, .bss, .rel.text (relocation info for the code section), .rel.data, .rel.rodata, .init (C++ global variable constructors), .fini (C++ global variable destructors), .symtab (symbol table), .dynsym (dynamic library symbol table), .strtab, .dynstr, .interp (interpreter path). Executable ELFs and relocatable ELFs are essentially identical in format; the data is simply rearranged so the file can be mapped directly into memory — i.e., pageable.

![sections_segments](https://149520725.v2.pressablecdn.com//wp-content/uploads/2018/01/Image5.png)


## Static Libraries and Dynamic Libraries
A static library is essentially a set of object files, with just a little bit of extra information (some systems even just concatenate object files and call that a legitimate static library).

After processing the regular input files, if the linker finds that some imported symbols are still undefined, it walks through the libraries, looks up those symbols, and links in the files that contain them.

Dynamic libraries complicate the linking process slightly, moving part of the above work from link time to load time. At link time, the linker finds the dynamic libraries that can resolve the undefined symbols, but does not link any code yet. Instead, it notes in the output file which dynamic library a given symbol can be found in, so that the loader binds those dynamic libraries when loading the program.

## Linking Must Follow the ABI
Every operating system provides an ABI for programs to use, covering system calls, techniques for wrapping system calls, memory address rules, register rules, and calling conventions. The linker must be ABI compliant: it must follow the ABI requirements, provide address tables for specific static data, and make standard function calls in a manner consistent with the calling convention.

Take Intel x86 as an example: it provides six 32-bit general-purpose registers EAX, EBX, ECX, EDX, ESI, EDI; two addressing registers EBP, ESP; and six 16-bit registers CS, DS, ES, FS, GS, SS. Among them, ESP is the hardware stack pointer, and EBP is usually the current frame pointer.

The x86 architecture has a hardware stack and a hardware return instruction — the hardware circuitry pushes the return address onto the stack and jumps to that address. Most other architectures keep it in a register instead, so on x86 the software does not need to save the return address from a register to somewhere in main memory.

[^1]: [Why an ELF executable could have 4 LOAD segments?](https://stackoverflow.com/questions/57761007/why-an-elf-executable-could-have-4-load-segments)
