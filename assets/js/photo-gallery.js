// Progressive loading for photo gallery pages.
//
// Photos pages embed many large JPEGs hosted on R2. render-image.html emits
// only the first image with a real `src`; all others sit inside a
// `.photo-frame--pending` placeholder with the real URL in `data-src`.
// This script fetches them on demand as they approach the viewport, with a
// concurrency cap so slow connections are not saturated by dozens of
// competing multi-MB downloads (the main cause of timeouts), plus
// stall detection, one automatic retry, and click-to-retry on failure.
//
// Downloads go through fetch() with a streamed response body so the
// placeholder can show live progress (percent of Content-Length). The
// completed bytes are handed to the <img> as a blob URL. If fetch is blocked
// (e.g. the R2 bucket lacks a CORS rule for this origin) the loader falls
// back to a plain <img> download — spinner only, no percentage.
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

  var pending = Array.prototype.slice.call(
    document.querySelectorAll('.photo-frame--pending')
  );
  if (!pending.length) return;

  var queue = [];
  var queued = new Set();
  var active = 0;

  function enqueue(frame, front) {
    if (queued.has(frame)) return;
    queued.add(frame);
    if (front) queue.unshift(frame);
    else queue.push(frame);
    pump();
  }

  function pump() {
    while (active < MAX_CONCURRENT && queue.length) {
      var frame = queue.shift();
      queued.delete(frame);
      load(frame);
    }
  }

  function load(frame) {
    var img = frame.querySelector('img');
    var status = frame.querySelector('.photo-frame__status');
    var url = img.getAttribute('data-src');
    var attempts = 0;
    active++;
    frame.classList.add('photo-frame--loading');

    frame.__retry = function () {
      frame.classList.remove('photo-frame--error');
      status.textContent = '';
      status.removeAttribute('role');
      enqueue(frame, true);
    };

    attempt();

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
        status.textContent = ''; // spinner only, no percentage
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
        if (ok) {
          status.textContent = '';
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
            status.textContent = MSG_ERROR;
            status.setAttribute('role', 'button');
          }
        }
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
                      URL.revokeObjectURL(objectUrl);
                      settle(true);
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
                  status.textContent = total
                    ? Math.round((loaded / total) * 100) + '%'
                    : (loaded / 1048576).toFixed(1) + ' MB';
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

  // Click a failed frame to retry.
  document.addEventListener('click', function (ev) {
    var frame =
      ev.target && ev.target.closest
        ? ev.target.closest('.photo-frame--error')
        : null;
    if (frame && frame.__retry) frame.__retry();
  });

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
    pending.forEach(function (frame) { io.observe(frame); });
  } else {
    pending.forEach(function (frame) { enqueue(frame, false); });
  }
})();
