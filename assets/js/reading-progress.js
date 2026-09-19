/**
 * reading-progress.js — a hairline reading indicator for writeup articles.
 *
 * A 2px line that traces the article's reading progress: the travelled part is
 * lit with a gradient, and a single spark pulses at the current position. It
 * reads like a scrollbar but stays quiet — no track, no chrome, and it only
 * fades in once the reader has actually started scrolling.
 *
 * Orientation decides the axis, matching which edge the shell puts a bar on:
 *   - landscape (>= md): vertical, riding the sidebar's right edge. The offset
 *     is measured from the sidebar itself rather than recomputed from the width
 *     tokens, so it follows the resizer and the collapsed state for free.
 *   - portrait (< md): horizontal, riding the subnav's bottom edge (below md the
 *     sidebar is gone and the subnav is the topmost bar).
 *
 * Only pages that are both a single page and a blog page get the line — that is
 * the writeups articles, not the section lists, photo galleries or tag pages.
 *
 * No-ops everywhere else.
 */
(function () {
  var body = document.body;
  if (!body.classList.contains('td-page') || !body.classList.contains('td-blog')) return;

  var el = document.createElement('div');
  el.className = 'td-read-progress';
  el.setAttribute('aria-hidden', 'true');
  el.innerHTML =
    '<span class="td-read-progress__fill"></span><span class="td-read-progress__spark"></span>';
  body.appendChild(el);

  var wide = window.matchMedia('(min-width: 768px)');
  var sidebar = document.getElementById('td-shell-sidebar');
  var ticking = false;

  // The travelled length is exposed as a plain 0–1 custom property; both the
  // gradient fill (scale) and the spark (position) read it in CSS, so the JS
  // never touches geometry per frame.
  function paint() {
    ticking = false;
    var max = document.documentElement.scrollHeight - window.innerHeight;
    var progress = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
    el.style.setProperty('--td-read-progress', progress);
    // Stay invisible until reading has actually begun, and once the end is
    // reached there is nothing left to suggest.
    el.classList.toggle('is-active', window.scrollY > 80 && progress < 0.999);
  }

  function measure() {
    if (wide.matches) {
      var width = sidebar ? sidebar.getBoundingClientRect().width : 0;
      el.style.setProperty('--td-read-progress-offset', width + 'px');
    } else {
      el.style.removeProperty('--td-read-progress-offset');
    }
    paint();
  }

  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(paint);
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', measure);
  wide.addEventListener('change', measure);
  if (sidebar && window.ResizeObserver) {
    // The sidebar is resizable and collapsible; follow it without polling.
    new ResizeObserver(measure).observe(sidebar);
  }

  measure();
})();
