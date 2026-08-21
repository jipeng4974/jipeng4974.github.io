/**
 * toc-sheet.js — phone-portrait behavior for the right TOC rail.
 *
 * Below md in portrait orientation the rail opens as a full-width top sheet
 * (~40dvh, see _styles_project.scss). This script handles the two behaviors
 * CSS cannot:
 *
 *   1. The collapsible groups (Tags, Actions) start collapsed so the sheet
 *      fits its height; "On this page" keeps the theme's keep-open exception
 *      and stays expanded. Mirrors the group toggling the theme's
 *      docs-shell.js applies when it relocates the aside into the drawer —
 *      disabled on this site, so it is re-done here for the sheet instead.
 *   2. A tap outside the sheet — on the backdrop covering the article's
 *      visible lower part, or anywhere else outside the panel — closes it.
 *      Tapping a TOC anchor inside the panel also closes it (the jump would
 *      otherwise land hidden behind the sheet).
 *
 * No-ops on pages without the rail and outside phone portrait.
 */
(function () {
  var aside = document.querySelector('[data-td-shell-aside]');
  if (!aside) return;
  var html = document.documentElement;
  var sheet = window.matchMedia('(max-width: 767.98px) and (orientation: portrait)');

  function setGroups(expanded) {
    aside
      .querySelectorAll('[data-td-shell-tree-toggle]:not([data-td-shell-aside-keep-open])')
      .forEach(function (button) {
        var target = document.getElementById(button.getAttribute('aria-controls'));
        if (!target) return;
        button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        target.classList.toggle('is-open', expanded);
        var label = expanded ? button.dataset.labelCollapse : button.dataset.labelExpand;
        if (label) button.setAttribute('aria-label', label);
      });
  }

  function apply() {
    setGroups(!sheet.matches);
  }
  apply();
  sheet.addEventListener('change', apply);

  function collapseRail() {
    html.setAttribute('data-td-shell-toc', 'collapsed');
    try {
      localStorage.setItem('td-shell-toc-collapsed', '1');
    } catch (e) {
      /* ignore */
    }
  }

  document.addEventListener('click', function (event) {
    if (!sheet.matches) return;
    if (html.getAttribute('data-td-shell-toc') === 'collapsed') return;
    if (event.target.closest('[data-td-shell-right-toggle]')) return;
    if (
      event.target.closest('.td-shell-toc-sheet-backdrop') ||
      event.target.closest('.td-shell-toc a[href^="#"]')
    ) {
      collapseRail();
    }
  });
})();
