#!/usr/bin/env python3
"""Generate the Traditional Chinese content tree from the Simplified pages.

The site keeps only ``content/**/*.md`` and ``content/**/*.zh.md`` in source
control.  The ``zh-trad`` language is a build-time transform of the latter, so
traditional text does not have to be maintained separately.  Output goes to
``tmp/content-zh-trad`` and is mounted into Hugo by ``hugo.yml``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "content"
OUTPUT = ROOT / "tmp" / "content-zh-trad"

# Executable code stays untouched so identifiers, examples, and inline code do
# not change under conversion.  Mermaid is prose-like and is localized.
CONVERT_CODE_LANGS = {"mermaid", "text", "txt"}
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)


def make_converter() -> Callable[[str], str]:
    """Return an OpenCC s2t function, using either the module or CLI."""
    try:
        from opencc import OpenCC

        return OpenCC("s2t").convert
    except ImportError:
        pass

    def convert_with_cli(text: str) -> str:
        result = subprocess.run(
            ["opencc", "-c", "s2t"],
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout

    return convert_with_cli


def convert_markdown(text: str, convert: Callable[[str], str]) -> str:
    """Convert Markdown prose, preserving executable code and inline code."""
    output: list[str] = []
    fence_marker = ""
    code_mode: str | None = None  # None outside a fence, "skip" or "convert"

    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if code_mode is None and match:
            marker, info = match.groups()
            language = info.strip().split(" ", 1)[0].strip("{}").lower()
            fence_marker = marker
            code_mode = "convert" if language in CONVERT_CODE_LANGS else "skip"
            output.append(line)
            continue

        if code_mode is not None:
            output.append(line if code_mode == "skip" else convert(line))
            if match and match.group(1).startswith(fence_marker):
                code_mode = None
            continue

        placeholders: dict[str, str] = {}

        def preserve(match: re.Match[str]) -> str:
            token = f"@@OPENCC_CODE_{len(placeholders)}@@"
            placeholders[token] = match.group(0)
            return token

        line_without_code = INLINE_CODE_RE.sub(preserve, line)
        converted = convert(line_without_code)
        for token, original in placeholders.items():
            converted = converted.replace(token, original)
        output.append(converted)

    return "".join(output)


def split_front_matter(text: str) -> tuple[str | None, str, str]:
    """Split the initial TOML or YAML front matter and return its delimiter."""
    match = re.match(r"\A(---|\+\+\+)[ \t]*\r?\n", text)
    if not match:
        return None, "", text

    delimiter = match.group(1)
    pattern = re.compile(
        rf"\A{re.escape(delimiter)}[ \t]*\r?\n(.*?)(?:\r?\n){re.escape(delimiter)}[ \t]*(?:\r?\n|$)",
        re.DOTALL,
    )
    block = pattern.match(text)
    if not block:
        return None, "", text

    return delimiter, block.group(1), text[block.end() :]


def convert_file(source: Path, target: Path, convert: Callable[[str], str]) -> None:
    text = source.read_text(encoding="utf-8")
    delimiter, front_matter, body = split_front_matter(text)

    if delimiter:
        body = convert_markdown(body, convert)
        front_matter = convert(front_matter)
        closing = "\r\n" if "\r\n" in text[:200] or "\r\n" in text[-200:] else "\n"
        text = f"{delimiter}\n{front_matter.rstrip()}\n{delimiter}{closing}{body}"
    else:
        text = convert_markdown(text, convert)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    convert = make_converter()
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    sources = sorted(SOURCE.rglob("*.zh.md"))
    if not sources:
        raise SystemExit("No content/**/*.zh.md files found")

    for source in sources:
        relative = source.relative_to(SOURCE)
        relative_trad = str(relative).removesuffix(".zh.md") + ".zh-trad.md"
        target = OUTPUT / relative_trad
        convert_file(source, target, convert)

    print(f"Generated {len(sources)} zh-trad pages in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
