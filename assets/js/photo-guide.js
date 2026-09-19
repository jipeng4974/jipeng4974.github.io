// Mosaic navigation for photograph pages.
//
// Hugo emits an ordered JSON list containing each remote photograph and its
// build-time aspect ratio. This script plans a rectangular guide above the
// full-size photo stream: dynamic programming partitions the ordered photos
// into rows, and each row is then assigned exact percentage widths. Rows with
// a smaller total aspect ratio become visually larger "hero" rows; denser
// rows use smaller canvases. The planner is deterministic for a given page,
// so reloading does not shuffle the topology.
(function () {
  'use strict';

  var MIN_CANVAS_ASPECT = 0.62; // extreme portraits are cropped, not stretched
  var MAX_CANVAS_ASPECT = 2.35; // extreme landscapes are cropped as well

  // The original-photo loader waits for this promise. It resolves after the
  // guide has finished requesting every thumbnail (success or failure), so
  // the cheap overview is not competing with dozens of multi-MB originals.
  var resolveThumbnails;
  var thumbnailTotal = 0;
  var thumbnailSettled = 0;
  window.photoGuideReady = new Promise(function (resolve) {
    resolveThumbnails = resolve;
  });

  // Display mode is decided by the viewport, not by configuration: portrait
  // screens (phones) read the legacy tiled flow; wider screens get the
  // sticky-card deck. The class here drives every photo-mode-* CSS rule.
  var stacked = window.innerWidth >= window.innerHeight;
  var contentRoot = document.querySelector('.td-content');
  if (contentRoot) {
    contentRoot.classList.add(
      stacked ? 'photo-mode-stacked' : 'photo-mode-mosaic'
    );
  }

  function settleThumbnail() {
    thumbnailSettled += 1;
    if (thumbnailSettled >= thumbnailTotal) resolveThumbnails();
  }

  function canonicalUrl(value) {
    try {
      var url = new URL(value, window.location.href);
      return url.origin + url.pathname;
    } catch (error) {
      return value;
    }
  }

  function pageSeed(text) {
    var hash = 2166136261;
    for (var index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function normalizeAspect(item) {
    var aspect = Number(item.aspect);
    if (!Number.isFinite(aspect) || aspect <= 0) return 1;
    return Math.min(MAX_CANVAS_ASPECT, Math.max(MIN_CANVAS_ASPECT, aspect));
  }

  // Small deterministic rotations imitate physical prints without making the
  // page look random after a reload.
  var CARD_ROTATIONS = [1, -1.2, 0.8, -0.65, 1.4, -0.9];

  function createStackedCards(cards) {
    if (!cards.length) return;
    var stack = document.createElement('div');
    stack.className = 'photo-stack';
    var firstWrapper = cards[0].wrapper;
    firstWrapper.parentNode.insertBefore(stack, firstWrapper);

    cards.forEach(function (card, index) {
      var anchor = document.createElement('div');
      anchor.className = 'photo-stack-anchor';
      stack.appendChild(anchor);
      stack.appendChild(card.wrapper);

      card.anchor = anchor;
      card.wrapper.__photoStackAnchor = anchor;
      card.wrapper.style.setProperty('--photo-aspect', String(card.aspect));
      card.wrapper.style.setProperty(
        '--photo-card-rotate',
        CARD_ROTATIONS[index % CARD_ROTATIONS.length] + 'deg'
      );
      card.wrapper.style.zIndex = String(index + 1);
    });
  }

  // More photographs get a smaller target canvas. The values are total row
  // aspect ratios: a desktop row near 3.4 holds few, large photographs while
  // one near 6.5 holds more, smaller photographs.
  function targetRowAspect(count) {
    if (count <= 12) return 3.3;
    if (count <= 40) return 4.3;
    if (count <= 80) return 5;
    return 5.6;
  }

  function normalizedPattern(seed) {
    // Low multipliers produce hero rows and high multipliers dense rows.
    var pattern = [0.72, 1, 1.2, 0.84, 1.08, 0.95];
    var sum = 0;
    var index;
    for (index = 0; index < pattern.length; index += 1) sum += pattern[index];
    for (index = 0; index < pattern.length; index += 1) {
      pattern[index] /= sum / pattern.length;
    }

    var rotation = seed % pattern.length;
    return pattern.slice(rotation).concat(pattern.slice(0, rotation));
  }

  function rowTargets(count, totalAspect, seed) {
    var base = targetRowAspect(count);
    var desiredRows = Math.max(1, Math.round(totalAspect / base));
    var rows = Math.min(count, desiredRows);
    var pattern = normalizedPattern(seed);
    var targets = [];
    var usedPatternSum = 0;
    var index;

    for (index = 0; index < rows; index += 1) {
      var multiplier = pattern[index % pattern.length];
      targets.push(multiplier);
      usedPatternSum += multiplier;
    }

    // Keep the mean target equal to base after rotation/truncation.
    var correction = totalAspect / usedPatternSum;
    for (index = 0; index < targets.length; index += 1) {
      targets[index] *= correction;
    }
    return targets;
  }

  // Partition the ordered photo list into exactly targets.length contiguous
  // rows. Minimizing squared deviation globally prevents the greedy algorithm
  // from leaving a lonely or oversized final row.
  function planRows(items, targets) {
    var photoCount = items.length;
    var rowCount = targets.length;
    var prefix = [0];
    var row;
    var photo;
    var previousPhoto;

    for (photo = 0; photo < photoCount; photo += 1) {
      prefix.push(prefix[photo] + normalizeAspect(items[photo]));
    }

    var costs = [];
    for (row = 0; row <= rowCount; row += 1) {
      costs.push(new Array(photoCount + 1).fill(Infinity));
    }
    costs[0][0] = 0;

    var parents = [];
    for (row = 0; row <= rowCount; row += 1) {
      parents.push(new Array(photoCount + 1).fill(-1));
    }

    for (row = 1; row <= rowCount; row += 1) {
      for (photo = row; photo <= photoCount - (rowCount - row); photo += 1) {
        for (previousPhoto = row - 1; previousPhoto < photo; previousPhoto += 1) {
          var previousCost = costs[row - 1][previousPhoto];
          if (!Number.isFinite(previousCost)) continue;

          var rowAspect = prefix[photo] - prefix[previousPhoto];
          var error = rowAspect - targets[row - 1];
          var cost = previousCost + error * error;
          if (cost < costs[row][photo]) {
            costs[row][photo] = cost;
            parents[row][photo] = previousPhoto;
          }
        }
      }
    }

    var ends = new Array(rowCount).fill(photoCount);
    var cursor = photoCount;
    for (row = rowCount; row >= 1; row -= 1) {
      ends[row - 1] = cursor;
      cursor = parents[row][cursor];
      if (cursor < 0) return [items.map(function (item) { return [item]; })];
    }

    var rows = [];
    var start = 0;
    for (row = 0; row < rowCount; row += 1) {
      rows.push(items.slice(start, ends[row]));
      start = ends[row];
    }
    return rows.filter(function (rowItems) { return rowItems.length > 0; });
  }

  function mainFrameFor(url, frameByUrl) {
    return frameByUrl.get(canonicalUrl(url)) || null;
  }

  function buildGuide(root, items) {
    var content = root.closest('.td-content');
    var frames = Array.prototype.slice.call(
      content.querySelectorAll('.photo-frame .photo-fit-screen')
    );
    if (!frames.length) {
      resolveThumbnails();
      return;
    }

    var frameByUrl = new Map();
    frames.forEach(function (frame) {
      if (frame.dataset.src) frameByUrl.set(canonicalUrl(frame.dataset.src), frame);
    });

    var usable = items
      .map(function (item) { return Object.assign({}, item); })
      .filter(function (item) { return mainFrameFor(item.url, frameByUrl); });
    if (!usable.length) {
      resolveThumbnails();
      return;
    }

    thumbnailTotal = usable.length;
    usable.forEach(function (item) {
      item.aspect = normalizeAspect(item);
      item.img = mainFrameFor(item.url, frameByUrl);
      item.wrapper = item.img.closest('.photo-frame');
    });

    if (stacked) createStackedCards(usable);
    var totalAspect = usable.reduce(function (sum, item) { return sum + item.aspect; }, 0);
    var targets = rowTargets(usable.length, totalAspect, pageSeed(window.location.pathname));
    var plannedRows = planRows(usable, targets);

    var grid = document.createElement('div');
    grid.className = 'photo-guide__grid';
    var buttons = [];

    plannedRows.forEach(function (rowItems) {
      var row = document.createElement('div');
      row.className = 'photo-guide__row';
      var rowAspect = rowItems.reduce(function (sum, item) { return sum + item.aspect; }, 0);

      rowItems.forEach(function (item) {
        var frame = mainFrameFor(item.url, frameByUrl);
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'photo-guide__tile';
        button.dataset.photoUrl = canonicalUrl(item.url);
        var isChinese = document.documentElement.lang.startsWith('zh');
        var jumpLabel = isChinese
          ? '跳转到#' + item.index + ' ' + item.stem
          : 'Jump to #' + item.index + ' ' + item.stem;
        button.title = jumpLabel;
        button.setAttribute('aria-label', jumpLabel);

        var image = document.createElement('img');
        image.alt = '';
        image.loading = 'eager';
        image.decoding = 'async';

        // Attach handlers before assigning src so a cache hit cannot finish
        // between assignment and listener registration.
        var preload = new Promise(function (resolve) {
          function finish(ok) {
            button.classList.remove(
              'photo-guide__img-loaded',
              'photo-guide__img-error'
            );
            button.classList.add(
              ok ? 'photo-guide__img-loaded' : 'photo-guide__img-error'
            );
            resolve();
          }

          image.addEventListener('load', function () { finish(true); });
          image.addEventListener('error', function () { finish(false); });
        });
        image.src = item.thumb;
        image.setAttribute('fetchpriority', 'high');
        preload.then(settleThumbnail);

        button.appendChild(image);
        button.style.aspectRatio = String(item.aspect);
        // Flex-grow is proportional to aspect ratio, so browser rounding
        // cannot make the row's right edge fall short of the rectangle.
        button.style.flexGrow = String(item.aspect);
        row.appendChild(button);
        buttons.push(button);
        item.button = button;
        button.__photoGuideItem = item;
      });

      grid.appendChild(row);
    });

    root.appendChild(grid);
    root.removeAttribute('hidden');

    var jumpAnimationId = 0;
    var jumpAnimationFrame = 0;

    function cancelJumpAnimation() {
      jumpAnimationId += 1;
      if (jumpAnimationFrame) {
        cancelAnimationFrame(jumpAnimationFrame);
        jumpAnimationFrame = 0;
      }
    }

    function animateGuideJump(targetY, targetIndex) {
      cancelJumpAnimation();
      var token = jumpAnimationId;
      var startY = window.scrollY;
      var distance = targetY - startY;

      // Keep jumps responsive: around 700ms for typical screen-sized moves,
      // with a little extra time for long jumps through the stack.
        // This is only the short photo-deck phase; the slow trip through the
        // thumbnail section is skipped by the instant relocation above.
        var duration = Math.min(
          700,
          Math.max(280, 220 + targetIndex * 55)
        );
      var startTime = performance.now();

      function easeInOutCubic(progress) {
        return progress < 0.5
          ? 4 * progress * progress * progress
          : 1 - Math.pow(-2 * progress + 2, 3) / 2;
      }

      function step(now) {
        if (token !== jumpAnimationId) return;
        var linear = Math.min(1, (now - startTime) / duration);
        var eased = easeInOutCubic(linear);
        window.scrollTo(0, startY + distance * eased);
        if (linear < 1) {
          jumpAnimationFrame = requestAnimationFrame(step);
        } else {
          jumpAnimationFrame = 0;
        }
      }

      jumpAnimationFrame = requestAnimationFrame(step);
    }

    function jumpToStackedCard(item, anchor) {
      function centeredStackPosition(card) {
        var wrapper = card.wrapper;
        var wrapperStyle = getComputedStyle(wrapper);
        var stickyTop = parseFloat(wrapperStyle.top) || 0;
      var slotMargin = parseFloat(wrapperStyle.marginTop) || 0;
      var slotHeight = anchor.offsetHeight || 0;
        var cardHeight = wrapper.offsetHeight;
        var anchorTop =
          card.anchor.getBoundingClientRect().top + window.scrollY;

        // An anchor is immediately before its card's 18svh slot margin. This
        // places the card's natural top at the viewport center; the sticky
        // threshold follows after a little further scrolling.
        var desiredCardTop = Math.max(
          stickyTop,
          (window.innerHeight - cardHeight) / 2
        );
        var maxScroll =
          document.documentElement.scrollHeight - window.innerHeight;
        return Math.min(
          maxScroll,
          Math.max(0, anchorTop + slotMargin + slotHeight - desiredCardTop)
        );
      }

      // Skip the guide section completely: snap to the beginning of the stack,
      // then run only the short card-deck animation to the target photograph.
      var startY = centeredStackPosition(usable[0]);
      var targetY = centeredStackPosition(item);
      window.scrollTo(0, startY);

      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        window.scrollTo(0, targetY);
      } else {
        animateGuideJump(targetY, usable.indexOf(item));
      }
    }

    root.addEventListener('click', function (event) {
      var button = event.target.closest('.photo-guide__tile');
      if (!button) return;

      var frame = mainFrameFor(button.dataset.photoUrl, frameByUrl);
      if (!frame) return;
      var wrapper = frame.closest('.photo-frame');
      var target = wrapper && wrapper.__photoStackAnchor
        ? wrapper.__photoStackAnchor
        : wrapper;
      if (!target) return;
      if (!target.id) {
        target.id = 'photo-' + button.dataset.photoUrl.split('/').pop().replace(/\.[^.]+$/, '');
      }

      if (stacked) {
        jumpToStackedCard(button.__photoGuideItem, target);
      } else {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      if (typeof window.photoGalleryPrioritize === 'function') {
        window.photoGalleryPrioritize(wrapper);
      }
      window.history.replaceState(null, '', '#' + target.id);
    });

    if (stacked) {
      // Sticky cards remain visible after their slot has passed, so viewport
      // intersection cannot identify the active photograph. Its invisible
      // flow anchor can.
      var activeCard = null;
      var activeUpdateScheduled = false;

      function updateStackedActive() {
        activeUpdateScheduled = false;
        var line = window.scrollY + Math.max(
          120,
          window.innerHeight * 0.35
        );
        var next = usable[0];

        usable.forEach(function (item) {
          if (!item.anchor) return;
          var top = item.anchor.getBoundingClientRect().top + window.scrollY;
          if (top <= line) next = item;
        });

        if (next && next !== activeCard) {
          activeCard = next;
          usable.forEach(function (item) {
            item.wrapper.classList.toggle(
              'photo-stack-card--active',
              item === next
            );
          });
        }
      }

      function scheduleStackedActiveUpdate() {
        if (activeUpdateScheduled) return;
        activeUpdateScheduled = true;
        requestAnimationFrame(updateStackedActive);
      }

      window.addEventListener('scroll', scheduleStackedActiveUpdate, {
        passive: true
      });
      window.addEventListener('resize', scheduleStackedActiveUpdate);
      requestAnimationFrame(updateStackedActive);
      window.addEventListener('wheel', cancelJumpAnimation, { passive: true });
      window.addEventListener('touchstart', cancelJumpAnimation, {
        passive: true
      });
    }
  }

  function initialize() {
    var root = document.querySelector('.photo-guide');
    var dataNode = root && root.querySelector('.photo-guide-data');
    if (!dataNode) {
      resolveThumbnails();
      return;
    }

    try {
      var items = JSON.parse(dataNode.textContent);
      // Some minifiers wrap inline application/json payloads in a string.
      if (typeof items === 'string') items = JSON.parse(items);
      buildGuide(root, items);
    } catch (error) {
      root.remove();
      resolveThumbnails();
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else {
    initialize();
  }
})();
