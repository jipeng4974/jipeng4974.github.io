// Auto-fit display math to the screen. KaTeX renders display equations with
// white-space: nowrap and never line-breaks, so wide equations overflow
// narrow (phone) screens. Every KaTeX dimension is em-based, so shrinking
// the font-size of the .katex root scales the whole equation uniformly.
// Each .katex-display is shrunk just enough to fit its container, down to a
// floor; anything still too wide scrolls horizontally (overflow fallback in
// _styles_project.scss).
(function () {
  "use strict";

  var MIN_SCALE = 0.55; // below this, horizontal scrolling takes over

  function fit(display) {
    var katex = display.firstElementChild; // .katex root
    if (!katex) return;
    katex.style.fontSize = ""; // restore natural size before measuring
    var available = display.clientWidth;
    var needed = display.scrollWidth;
    if (!available || needed <= available) return;
    var base = parseFloat(window.getComputedStyle(katex).fontSize);
    if (!base) return;
    var scale = Math.max(available / needed, MIN_SCALE);
    katex.style.fontSize = base * scale + "px";
  }

  function fitAll() {
    document.querySelectorAll(".katex-display").forEach(fit);
  }

  var pending = null;
  window.addEventListener("resize", function () {
    if (pending !== null) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(fitAll);
  });

  fitAll();
  // Refit once KaTeX web fonts finish loading — metrics shift slightly.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitAll);
})();
