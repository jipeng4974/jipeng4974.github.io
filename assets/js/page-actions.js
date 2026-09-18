/**
 * page-actions.js — "Copy link" and "Copy BibTeX" in the TOC rail's Actions.
 *
 * The theme's docs-shell.js already handles "Copy Markdown", but that handler
 * is built around fetching the page's Markdown output from a data-url and lives
 * in a closure, so these two string copies get their own small module. The
 * markup (layouts/_partials/page-meta-links.html) deliberately reuses the
 * theme's feedback contract, so both feel identical:
 *
 *   - .is-copied on the button swaps its icon for a check (theme CSS);
 *   - the [data-td-page-copy-label] span flashes the button's data-t-copied
 *     text, then restores the original label;
 *   - the group's aria-live [data-td-page-context-status] announces the same
 *     message for screen readers.
 *
 * Keep those two implementations in step if the contract changes.
 *
 * The BibTeX entry is assembled from the button's data-bibtex-* attributes,
 * which the partial fills from params.citation in hugo.yml plus the page's
 * title, date and permalink. Undated pages (section indexes, the home page)
 * ship an empty year/month and those fields are dropped.
 *
 * No-ops on pages without the rail's Actions group.
 */
(function () {
  var buttons = document.querySelectorAll('[data-td-copy-link], [data-td-copy-bibtex]');
  if (!buttons.length) return;

  var FEEDBACK_MS = 1400;

  function fallbackCopy(text) {
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      try {
        if (document.execCommand('copy')) resolve();
        else reject(new Error('copy failed'));
      } catch (error) {
        reject(error);
      }
      textarea.remove();
    });
  }

  function writeClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return fallbackCopy(text);
  }

  // @online{<key>,
  //   title  = {…},
  //   author = {…},
  //   year   = {…},
  //   month  = {…},
  //   url    = {\url{…}},
  // }
  function bibtex(button) {
    var d = button.dataset;
    var lines = ['@online{' + d.bibtexKey + ','];
    if (d.bibtexTitle) lines.push('  title  = {' + d.bibtexTitle + '},');
    if (d.bibtexAuthor) lines.push('  author = {' + d.bibtexAuthor + '},');
    if (d.bibtexYear) lines.push('  year   = {' + d.bibtexYear + '},');
    if (d.bibtexMonth) lines.push('  month  = {' + d.bibtexMonth + '},');
    lines.push('  url    = {\\url{' + d.bibtexUrl + '}},');
    lines.push('}');
    return lines.join('\n');
  }

  function flash(button, ok) {
    var root = button.closest('[data-td-page-context]');
    var label = button.querySelector('[data-td-page-copy-label]');
    var status = root && root.querySelector('[data-td-page-context-status]');
    var message =
      (ok ? button.dataset.tCopied : button.dataset.tCopyError) ||
      (ok ? 'Copied' : 'Could not copy');

    if (ok) button.classList.add('is-copied');
    if (label && !label.dataset.original) label.dataset.original = label.textContent;
    if (label) label.textContent = message;
    if (status) {
      status.textContent = '';
      window.requestAnimationFrame(function () {
        status.textContent = message;
      });
    }
    window.setTimeout(function () {
      button.classList.remove('is-copied');
      if (label && label.dataset.original) label.textContent = label.dataset.original;
    }, FEEDBACK_MS);
  }

  Array.prototype.forEach.call(buttons, function (button) {
    button.addEventListener('click', function () {
      // "Copy link" copies what the address bar shows (anchors included);
      // "Copy BibTeX" builds the entry from the button's data attributes.
      var text = button.hasAttribute('data-td-copy-link')
        ? window.location.href
        : bibtex(button);
      writeClipboard(text).then(
        function () {
          flash(button, true);
        },
        function () {
          flash(button, false);
        },
      );
    });
  });
})();
