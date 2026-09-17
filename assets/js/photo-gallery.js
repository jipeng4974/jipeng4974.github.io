// Progressive loading for photo gallery pages.
//
// Photos pages embed many large JPEGs hosted on R2. render-image.html wraps
// every image in a `.photo-frame--pending` placeholder with the real URL in
// `data-src`, and this script fetches them on demand as they approach the
// viewport, with a concurrency cap so slow connections are not saturated by
// dozens of competing multi-MB downloads (the main cause of timeouts), plus
// stall detection, one automatic retry, and click-to-retry on failure.
//
// Downloads go through fetch() with a streamed response body so the
// placeholder can show live progress (percent of Content-Length). The
// completed bytes are handed to the <img> as a blob URL. If fetch is blocked
// (e.g. the R2 bucket lacks a CORS rule for this origin) the loader falls
// back to a plain <img> download — spinner only, no percentage.
//
// A lightbox is layered on top: clicking any photo opens it fullscreen with
// prev/next navigation (wrap-around), ESC / X / backdrop-click to close, and
// the same live progress display while the photo is still downloading
// (clicking a pending photo prioritizes it ahead of the prefetch queue).
(function () {
  'use strict';

  var MAX_CONCURRENT = 2;
  var STALL_TIMEOUT_MS = 45000; // no bytes for this long -> treated as failed
  var MAX_AUTO_RETRIES = 1;
  var PLACEHOLDER_SRC =
    'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

  var zh = /^zh\b/i.test(document.documentElement.lang || '');
  var MSG_ERROR = zh
    ? '加载失败，点击重试'
    : 'Failed to load — click to retry';
  // Washi-tape tag states: "0%" (server-rendered initial) -> percent ->
  // "#N" once loaded. Minimal by design; the photo itself is the content.
  var TAG_RESTART = '0%'; // reset after a retry
  var TAG_INDETERMINATE = '…'; // streamed progress unavailable (plainLoad)
  var TAG_FAILED = '×'; // brief on-card mark; the frame click retries

  // Every photo on the page, in display order; each lives inside a
  // placeholder frame rendered by layouts/_markup/render-image.html.
  var photos = Array.prototype.slice.call(
    document.querySelectorAll('.td-content .photo-fit-screen')
  );
  var pending = photos.filter(function (img) {
    return img.closest('.photo-frame--pending');
  });
  if (!photos.length) return;

  var queue = [];
  var queued = new Set();
  var active = 0;

  function frameOf(img) {
    return img.closest('.photo-frame');
  }

  function tagOf(frame) {
    return frame.querySelector('.photo-frame__tag');
  }

  function setTagText(frame, text) {
    var tag = tagOf(frame);
    if (tag) tag.textContent = text;
  }

  // "#6" — the photo's position on the page, stamped onto the tag by
  // render-image.html.
  function tagLoadedText(frame) {
    var tag = tagOf(frame);
    var n = parseInt(tag && tag.getAttribute('data-photo-index'), 10);
    return n > 0 ? '#' + n : '';
  }

  // Tear the tag off a loaded photo: it splits into two halves; the left
  // half gets a leftward force, the right half a rightward one, and both
  // tumble down off-screen, leaving a clean photo card. Trajectory, spin,
  // speed, and stagger are randomized per click so no two tears look the
  // same. The halves are fixed-position clones on <body>, because the card
  // clips its overflow.
  function shatterTag(frame, tag) {
    if (frame.__tagShattered) return;
    frame.__tagShattered = true;

    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    var width = tag.offsetWidth;
    var height = tag.offsetHeight;
    var rect = tag.getBoundingClientRect();
    var centerX = rect.left + rect.width / 2;
    var centerY = rect.top + rect.height / 2;

    // Hide the original only after measuring, and clone fresh copies for
    // the falling halves (the original carries visibility:hidden, so each
    // clone must opt back in explicitly).
    tag.style.visibility = 'hidden';

    // Each half is clipped in the tag's local box (before the 45deg
    // rotation), so the split runs exactly down the band's long axis.
    var halves = [
      { clip: 'inset(0 50% 0 0)', dir: -1 }, // flies left
      { clip: 'inset(0 0 0 50%)', dir: 1 }   // flies right
    ];

    halves.forEach(function (half, index) {
      var piece = tag.cloneNode(true);
      // The piece leaves the frame's DOM, so the scoped .photo-frame__tag
      // rules stop matching; .photo-tag-piece (same block in SCSS) restores
      // the tag's visual identity at its fixed page position.
      piece.classList.add('photo-tag-piece');
      piece.style.visibility = 'visible';
      piece.style.position = 'fixed';
      piece.style.left = (centerX - width / 2) + 'px';
      piece.style.top = (centerY - height / 2) + 'px';
      piece.style.width = width + 'px';
      piece.style.height = height + 'px';
      piece.style.margin = '0';
      piece.style.clipPath = half.clip;
      piece.style.zIndex = '2000'; // over the page, under the lightbox
      piece.style.pointerEvents = 'none';
      document.body.appendChild(piece);

      // Randomized physics: horizontal gust, drop distance, tumble, and
      // speed all vary; the second half starts a beat later, as if the
      // tear propagates through the paper.
      var drift = half.dir * (24 + Math.random() * 18);      // vw
      var drop = 110 + Math.random() * 20;                   // vh
      var spin = half.dir * (120 + Math.random() * 100);     // deg
      var duration = 800 + Math.random() * 300;              // ms
      var delay = index * 50 + Math.random() * 60;           // ms; tear propagates

      var animation = piece.animate(
        [
          { transform: 'translate(0, 0) rotate(-45deg) rotate(0deg)' },
          {
            // translate() leads the list, so the gust/drop act in screen
            // space (gravity straight down, drift straight sideways) and
            // only the piece's own orientation is rotated.
            transform: 'translate(' + drift +
              'vw, ' + drop + 'vh) rotate(-45deg) rotate(' + spin + 'deg)'
          }
        ],
        {
          duration: duration,
          delay: Math.abs(delay),
          // Accelerating like gravity: slow tear, fast drop.
          easing: 'cubic-bezier(0.45, 0, 0.85, 0.5)',
          fill: 'forwards'
        }
      );
      animation.onfinish = function () { piece.remove(); };
      animation.oncancel = function () { piece.remove(); };
    });
  }

  // Every pending card shows its thumbnail immediately. These are the exact
  // URLs the guide mosaic loads, so the HTTP cache serves them without a
  // second download. The thumb paints below the original <img>, which stays
  // transparent until its own bytes have arrived.
  pending.forEach(function (img) {
    var frame = frameOf(img);
    var thumbUrl = frame.getAttribute('data-thumb');
    if (!thumbUrl || frame.querySelector('.photo-frame__thumb')) return;
    var thumb = document.createElement('img');
    thumb.className = 'photo-frame__thumb';
    thumb.alt = '';
    thumb.setAttribute('aria-hidden', 'true');
    thumb.decoding = 'async';
    thumb.src = thumbUrl;
    frame.insertBefore(thumb, img); // DOM order: thumb paints below original
  });

  function enqueue(frame, front) {
    if (queued.has(frame) || frame.__loading) return;
    if (frame.classList.contains('photo-frame--loaded')) return;
    queued.add(frame);
    if (front) queue.unshift(frame);
    else queue.push(frame);
    pump();
  }

  function pump() {
    while (active < MAX_CONCURRENT && queue.length) {
      var frame = queue.shift();
      queued.delete(frame);
      if (frame.__loading) continue;
      load(frame);
    }
  }

  // User explicitly asked for this photo (opened it in the lightbox):
  // pull it out of the prefetch queue and start it immediately, even if the
  // concurrency cap is already saturated by background prefetches.
  function prioritize(frame) {
    if (frame.__loading) return;
    if (frame.classList.contains('photo-frame--loaded')) return;
    var qi = queue.indexOf(frame);
    if (qi !== -1) {
      queue.splice(qi, 1);
      queued.delete(frame);
    }
    load(frame);
  }

  // The thumbnail guide can promote a jump target ahead of the background
  // queue; otherwise a late guide click could wait behind dozens of photos.
  window.photoGalleryPrioritize = prioritize;

  function load(frame) {
    // The frame now also contains a .photo-frame__thumb before the original,
    // so select the original explicitly instead of the first <img>.
    var img = frame.querySelector('.photo-fit-screen');
    var url = img.getAttribute('data-src');
    var attempts = 0;
    active++;
    frame.__loading = true;
    frame.classList.add('photo-frame--loading');

    frame.__retry = function () {
      frame.classList.remove('photo-frame--error');
      setTagText(frame, TAG_RESTART);
      enqueue(frame, true);
    };

    attempt();

    function notify() {
      if (frame.__notify) frame.__notify();
    }

    function attempt() {
      var settled = false;
      var stallTimer = setTimeout(onStall, STALL_TIMEOUT_MS);
      var controller =
        'AbortController' in window ? new AbortController() : null;
      // Retries get a cache-busting query so a poisoned/partial cache entry
      // or a dead connection is not reused.
      var src = attempts
        ? url + (url.indexOf('?') === -1 ? '?' : '&') + 'retry=' + attempts
        : url;

      // Reset the stall watchdog; called on every received chunk.
      function rearm() {
        clearTimeout(stallTimer);
        stallTimer = setTimeout(onStall, STALL_TIMEOUT_MS);
      }

      function plainLoad() {
        setTagText(frame, TAG_INDETERMINATE); // no streamed progress
        img.onload = function () { settle(true); };
        img.onerror = function () { settle(false); };
        img.src = src;
      }

      function settle(ok) {
        if (settled) return;
        settled = true;
        clearTimeout(stallTimer);
        img.onload = img.onerror = null;
        active--;
        frame.__loading = false;
        if (ok) {
          setTagText(frame, tagLoadedText(frame));
          frame.classList.remove(
            'photo-frame--pending',
            'photo-frame--loading',
            'photo-frame--error'
          );
          frame.classList.add('photo-frame--loaded');
        } else {
          attempts++;
          if (attempts <= MAX_AUTO_RETRIES) {
            enqueue(frame, true); // automatic retry, ahead of the queue
          } else {
            frame.classList.remove('photo-frame--loading');
            frame.classList.add('photo-frame--error');
            setTagText(frame, TAG_FAILED);
          }
        }
        notify();
        pump();
      }

      function onStall() {
        if (settled) return;
        if (controller) controller.abort();
        img.onload = img.onerror = null;
        img.src = PLACEHOLDER_SRC; // cancels any in-flight <img> request
        settle(false);
      }

      if (window.fetch && controller) {
        fetch(src, { signal: controller.signal }).then(
          function (resp) {
            if (settled) return;
            if (!resp.ok || !resp.body || !resp.body.getReader) {
              plainLoad();
              return;
            }
            var total = parseInt(resp.headers.get('Content-Length'), 10) || 0;
            var reader = resp.body.getReader();
            var chunks = [];
            var loaded = 0;
            (function read() {
              reader.read().then(
                function (r) {
                  if (settled) return;
                  if (r.done) {
                    var objectUrl = URL.createObjectURL(new Blob(chunks));
                    chunks = null;
                    img.onload = function () {
                      settle(true);
                      // The blob URL is only a way to display the bytes we
                      // just streamed. Swap it for the canonical URL (served
                      // from the HTTP cache, so instant) once it can be
                      // preloaded, so "open image in new tab" and the lightbox
                      // use the real address. Only then revoke the blob.
                      var swap = new Image();
                      swap.onload = function () {
                        img.src = url;
                        URL.revokeObjectURL(objectUrl);
                        notify(); // let the lightbox pick up the canonical src
                      };
                      swap.onerror = function () {
                        // Keep the (still valid) blob URL on display.
                      };
                      swap.src = url;
                    };
                    img.onerror = function () {
                      URL.revokeObjectURL(objectUrl);
                      settle(false);
                    };
                    img.src = objectUrl;
                    return;
                  }
                  chunks.push(r.value);
                  loaded += r.value.length;
                  rearm();
                  setTagText(frame, total
                    ? Math.round((loaded / total) * 100) + '%'
                    : (loaded / 1048576).toFixed(1) + ' MB');
                  notify();
                  read();
                },
                function () { settle(false); } // stream broke mid-transfer
              );
            })();
          },
          function () {
            // fetch itself rejected (CORS, offline, ...): degrade to a plain
            // <img> download so the photo still loads, just without progress.
            if (!settled) plainLoad();
          }
        );
      } else {
        plainLoad();
      }
    }
  }

  // ---------------------------------------------------------------------
  // Lightbox
  // ---------------------------------------------------------------------

  var lb = null; // lazily built DOM refs

  function ensureLightbox() {
    if (lb) return lb;
    var overlay = document.createElement('div');
    overlay.className = 'photo-lightbox';
    overlay.setAttribute('hidden', '');
    overlay.innerHTML =
      '<div class="photo-lightbox__status">' +
      '<span class="photo-lightbox__spinner"></span>' +
      '<span class="photo-lightbox__progress"></span>' +
      '</div>' +
      '<img class="photo-lightbox__img" alt="">' +
      '<button type="button" class="photo-lightbox__btn photo-lightbox__close" aria-label="Close">&#215;</button>' +
      '<button type="button" class="photo-lightbox__btn photo-lightbox__prev" aria-label="Previous">&#8249;</button>' +
      '<button type="button" class="photo-lightbox__btn photo-lightbox__next" aria-label="Next">&#8250;</button>';
    document.body.appendChild(overlay);

    lb = {
      overlay: overlay,
      img: overlay.querySelector('.photo-lightbox__img'),
      status: overlay.querySelector('.photo-lightbox__status'),
      spinner: overlay.querySelector('.photo-lightbox__spinner'),
      progress: overlay.querySelector('.photo-lightbox__progress'),
      idx: 0,
      watching: null, // frame whose __notify we currently own
    };

    overlay.querySelector('.photo-lightbox__close').addEventListener(
      'click',
      closeLightbox
    );
    overlay.querySelector('.photo-lightbox__prev').addEventListener(
      'click',
      function () { showPhoto(lb.idx - 1); }
    );
    overlay.querySelector('.photo-lightbox__next').addEventListener(
      'click',
      function () { showPhoto(lb.idx + 1); }
    );
    overlay.addEventListener('click', function (ev) {
      if (ev.target === overlay) closeLightbox(); // backdrop click
    });
    // A failed photo shows its retry hint inside the lightbox too.
    lb.status.addEventListener('click', function () {
      var img = photos[lb.idx];
      var frame = img && frameOf(img);
      if (
        frame &&
        frame.classList.contains('photo-frame--error') &&
        frame.__retry
      ) {
        frame.__retry();
        syncLightbox();
      }
    });
    document.addEventListener('keydown', function (ev) {
      if (lb.overlay.hasAttribute('hidden')) return;
      if (ev.key === 'Escape') closeLightbox();
      else if (ev.key === 'ArrowLeft') showPhoto(lb.idx - 1);
      else if (ev.key === 'ArrowRight') showPhoto(lb.idx + 1);
    });
    return lb;
  }

  function openLightbox(idx) {
    ensureLightbox();
    lb.overlay.removeAttribute('hidden');
    document.body.style.overflow = 'hidden';
    showPhoto(idx);
  }

  function closeLightbox() {
    lb.overlay.setAttribute('hidden', '');
    document.body.style.overflow = '';
    unwatch();
  }

  function unwatch() {
    if (lb && lb.watching) {
      lb.watching.__notify = null;
      lb.watching = null;
    }
  }

  function showPhoto(idx) {
    // Wrap around both ends.
    lb.idx = ((idx % photos.length) + photos.length) % photos.length;
    var img = photos[lb.idx];
    var frame = frameOf(img);

    unwatch();
    frame.__notify = syncLightbox;
    lb.watching = frame;
    if (!frame.classList.contains('photo-frame--loaded')) prioritize(frame);
    syncLightbox();
  }

  function syncLightbox() {
    if (!lb || lb.overlay.hasAttribute('hidden')) return;
    var img = photos[lb.idx];
    var frame = frameOf(img);
    var loaded = frame.classList.contains('photo-frame--loaded');

    if (loaded) {
      // Use .src (the assigned value) rather than .currentSrc, which lags
      // behind during the blob -> canonical swap.
      lb.img.src = img.src;
      lb.img.removeAttribute('hidden');
      lb.status.setAttribute('hidden', '');
      lb.progress.textContent = '';
    } else {
      lb.img.setAttribute('hidden', '');
      lb.status.removeAttribute('hidden');
      var failed = frame.classList.contains('photo-frame--error');
      lb.spinner.style.display = failed ? 'none' : '';
      lb.status.style.cursor = failed ? 'pointer' : '';
      var tag = tagOf(frame);
      lb.progress.textContent = failed
        ? MSG_ERROR
        : (tag ? tag.textContent : '');
    }
  }

  // ---------------------------------------------------------------------
  // Global click handling: retry failed frames, open the lightbox otherwise.
  // ---------------------------------------------------------------------

  document.addEventListener('click', function (ev) {
    var el =
      ev.target && ev.target.closest ? ev.target : null;
    if (!el) return;
    if (lb && !lb.overlay.hasAttribute('hidden')) return; // lightbox handles its own clicks

    var errFrame = el.closest('.photo-frame--error');
    if (errFrame) {
      if (errFrame.__retry) errFrame.__retry();
      return;
    }
    var img = el.closest('.td-content .photo-fit-screen');
    if (img) {
      var idx = photos.indexOf(img);
      if (idx !== -1) openLightbox(idx);
    }
  });

  // Sticky-note tag clicks: tear the tag off once the original photo has
  // loaded. While loading or on error, the card's own handlers keep
  // control (progress display / retry hint).
  document.addEventListener('click', function (ev) {
    var tag = ev.target && ev.target.closest
      ? ev.target.closest('.photo-frame__tag')
      : null;
    if (!tag) return;
    var frame = tag.closest('.photo-frame');
    if (!frame || !frame.classList.contains('photo-frame--loaded')) return;
    ev.stopPropagation();
    shatterTag(frame, tag);
  });

  // The thumbnail guide creates window.photoGuideReady before this script
  // runs. Do not start the expensive original-photo queue until it resolves,
  // so the small overview gets first use of the available bandwidth.
  var originalsStarted = false;
  function startOriginalQueue() {
    if (originalsStarted) return;
    originalsStarted = true;

    if ('IntersectionObserver' in window) {
      // Start fetching one viewport ahead of the scroll position.
      var io = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              io.unobserve(entry.target);
              enqueue(entry.target, false);
            }
          });
        },
        { rootMargin: '100% 0px' }
      );
      pending.forEach(function (img) {
        io.observe(img.closest('.photo-frame'));
      });
    } else {
      pending.forEach(function (img) {
        enqueue(img.closest('.photo-frame'), false);
      });
    }
  }

  if (window.photoGuideReady instanceof Promise) {
    window.photoGuideReady.then(startOriginalQueue, startOriginalQueue);
  } else {
    startOriginalQueue();
  }
})();
