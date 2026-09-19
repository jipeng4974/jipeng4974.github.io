// Writeup drop cap: promotes the first letter of the first prose paragraph on
// every page (Chinese pages simply treat the first Han character as that
// letter) to a two-line floating initial. Chinese inks and faces are randomised
// per page load; inline code and KaTeX runs are barriers, so the script bails
// rather than floating text past them.
(() => {
  const LETTER = /[\p{L}]/u;
  const ALNUM = /[\p{L}\p{N}]/u;
  const LATIN = /^[A-Za-z\u00c0-\u024f]$/;
  const isWhitespace = (ch) => /\s/u.test(ch);

  // Muted traditional inks: cinnabar, ochre, rouge, dark crimson, rosewood,
  // mineral blue/green, dai gray-blue, violet and ink black. All stay
  // low-saturation enough for the warm paper body.
  const INKS = [
    "#b3261e",
    "#9c5a3c",
    "#9e4f5f",
    "#7a3548",
    "#6b4a5a",
    "#4f6d7a",
    "#4f6f5e",
    "#4b5c6b",
    "#5d4a72",
    "#3f3a36",
  ];

  // Chinese face variants: kai, song, hei, light serif, black serif, mono.
  // The stacks degrade to Noto CJK families on Linux while still changing
  // family/weight.
  const FACES = [
    "td-dropcap--kai",
    "td-dropcap--song",
    "td-dropcap--hei",
    "td-dropcap--light",
    "td-dropcap--black",
    "td-dropcap--mono",
  ];

  const pick = (items, key) => {
    let previous = null;
    try {
      previous = sessionStorage.getItem(key);
    } catch (_) {
      // sessionStorage can be unavailable in privacy modes; random is enough.
    }

    let index = Math.floor(Math.random() * items.length);
    if (items.length > 1 && String(index) === previous) {
      index = (index + 1 + Math.floor(Math.random() * (items.length - 1))) % items.length;
    }

    try {
      sessionStorage.setItem(key, String(index));
    } catch (_) {
      // Ignore storage failures.
    }

    return items[index];
  };

  const isIgnored = (node) => {
    const parent = node.parentElement;
    return (
      !parent ||
      !!parent.closest("code, pre, .katex, script, style, .pog-pop-clone")
    );
  };

  const collect = (root) => {
    const chars = [];
    const barriers = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;

    while ((node = walker.nextNode())) {
      const text = node.textContent;
      if (isIgnored(node)) {
        if (text.trim()) barriers.push(chars.length);
        continue;
      }

      let offset = 0;
      for (const ch of text) {
        chars.push({ node, offset, ch });
        offset += ch.length;
      }
    }

    return { chars, barriers };
  };

  const findTarget = (root) => {
    const { chars, barriers } = collect(root);
    const first = chars.findIndex((c) => LETTER.test(c.ch));
    if (first < 0) return null;
    if (barriers.some((index) => index <= first)) return null;

    // Include leading punctuation / symbols in the float, but never numbers or
    // other letters, so the initial is a genuine prefix of the paragraph.
    let start = first;
    while (start > 0 && !ALNUM.test(chars[start - 1].ch)) {
      start -= 1;
    }
    if (chars.slice(0, start).some((c) => !isWhitespace(c.ch))) {
      return null;
    }
    while (start < first && isWhitespace(chars[start].ch)) {
      start += 1;
    }

    // Always a single initial: the first Han character on Chinese pages, the
    // first Latin letter on English pages.
    const end = first;

    return {
      start: chars[start],
      end: chars[end],
      isLatin: LATIN.test(chars[first].ch),
    };
  };

  const wrap = ({ start, end, isLatin }) => {
    const range = document.createRange();
    range.setStart(start.node, start.offset);
    range.setEnd(end.node, end.offset + end.ch.length);

    const span = document.createElement("span");
    span.className = "td-dropcap";
    if (isLatin) {
      span.classList.add("td-dropcap--latin");
    } else {
      span.classList.add(pick(FACES, "td-dropcap-face"));
      span.style.setProperty("--td-dropcap-ink", pick(INKS, "td-dropcap-ink"));
      span.style.setProperty(
        "--td-dropcap-scale",
        (0.94 + Math.random() * 0.14).toFixed(3)
      );
    }
    span.appendChild(range.extractContents());
    range.insertNode(span);
  };

  const init = () => {
    if (
      !document.body.classList.contains("td-page") ||
      !document.body.classList.contains("td-blog")
    ) {
      return;
    }

    for (const paragraph of document.querySelectorAll(".td-content > p")) {
      const target = findTarget(paragraph);
      if (target) {
        wrap(target);
        return;
      }
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
