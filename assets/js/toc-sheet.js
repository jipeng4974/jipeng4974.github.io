/**
 * toc-sheet.js — phone-portrait behavior for the right TOC rail.
 *
 * Below md in portrait orientation the rail opens as a full-width top sheet
 * (~40dvh, see _styles_project.scss). This script handles the behaviors CSS
 * cannot:
 *
 *   1. The collapsible groups (Tags, Actions) start collapsed so the sheet
 *      fits its height; "Contents" keeps its keep-open exception only on pages
 *      that actually have a heading tree, so it opens by default there and
 *      stays shut on section pages (whose contents group lists child pages
 *      instead — layouts/_partials/shell/section-pages.html). Mirrors the group
 *      toggling the theme's docs-shell.js applies when it relocates the aside
 *      into the drawer — disabled on this site, so it is re-done here for the
 *      sheet instead.
 *   2. The sheet's tab row (toc-aside.html): one button per group, acting as an
 *      accordion — the pressed one opens and the others close. The buttons are
 *      hidden outside the sheet, so the rail keeps its stacked groups on
 *      desktop and landscape, where this whole module leaves them expanded.
 *   3. A tap outside the sheet — on the backdrop covering the article's
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
  var tabButtons = document.querySelectorAll('[data-td-sheet-tab]');

  function setGroup(button, target, expanded) {
    if (!button || !target) return;
    button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    target.classList.toggle('is-open', expanded);
    var label = expanded ? button.dataset.labelCollapse : button.dataset.labelExpand;
    if (label) button.setAttribute('aria-label', label);
  }

  function setGroups(expanded) {
    aside
      .querySelectorAll('[data-td-shell-tree-toggle]:not([data-td-shell-aside-keep-open])')
      .forEach(function (button) {
        setGroup(button, document.getElementById(button.getAttribute('aria-controls')), expanded);
      });
  }

  // One group per sheet tab. The taxonomy group builds its own id, so it is
  // looked up by class rather than by the group's heading anchor.
  function groupFor(name) {
    if (name === 'toc') return aside.querySelector('.td-shell-aside-group--toc');
    if (name === 'tags') return aside.querySelector('.td-shell-tags-cloud');
    if (name === 'actions') {
      var body = document.getElementById('td-shell-aside-actions');
      return body ? body.closest('.td-shell-aside-group') : null;
    }
    return null;
  }

  function groupBody(group) {
    return group ? group.querySelector('.td-shell-tree__children') : null;
  }

  // Keep the row in step with the groups, whoever toggled them.
  function syncTabs() {
    Array.prototype.forEach.call(tabButtons, function (tab) {
      var group = groupFor(tab.dataset.tdSheetTab);
      var body = groupBody(group);
      // Pages without a heading tree and without child pages render no
      // contents group at all: drop its tab instead of leaving a dead button.
      tab.hidden = !body;
      if (!body) return;
      var open = body.classList.contains('is-open');
      tab.setAttribute('aria-expanded', open ? 'true' : 'false');
      tab.classList.toggle('is-active', open);
      if (body.id) tab.setAttribute('aria-controls', body.id);
    });
  }

  function apply() {
    setGroups(!sheet.matches);
    syncTabs();
  }
  apply();
  sheet.addEventListener('change', apply);

  Array.prototype.forEach.call(tabButtons, function (tab) {
    tab.addEventListener('click', function () {
      if (!sheet.matches) return;
      var group = groupFor(tab.dataset.tdSheetTab);
      var body = groupBody(group);
      if (!body) return;
      var willOpen = !body.classList.contains('is-open');
      ['toc', 'tags', 'actions'].forEach(function (name) {
        var other = groupFor(name);
        if (other) {
          setGroup(other.querySelector('[data-td-shell-tree-toggle]'), groupBody(other), false);
        }
      });
      if (willOpen) {
        setGroup(group.querySelector('[data-td-shell-tree-toggle]'), body, true);
      }
      syncTabs();
    });
  });

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
