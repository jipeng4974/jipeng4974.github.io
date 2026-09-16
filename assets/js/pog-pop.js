// Pog pop: clicking the brand logo plays a grow-then-shrink overlay
// animation on a fixed-position clone, so the layout never shifts. The
// clone is the logo SVG inlined, so the eyes can morph from tall ovals to
// round "surprised" circles in sync with the face growing.
(() => {
  const SELECTOR = ".td-shell-sidebar__brand img, .td-shell-subnav__brand img";
  const SCALE = 6;
  const GROW_MS = 500;
  const SHRINK_MS = 900;
  const DURATION = GROW_MS + SHRINK_MS;
  const PEAK = GROW_MS / DURATION;
  // Eyes go from 57x85 ovals to 72x72 rounds at full surprise.
  const EYE_SCALE_X = 72 / 57;
  const EYE_SCALE_Y = 72 / 85;

  let svgTextPromise = null;
  const loadSvgText = (src) => {
    if (!svgTextPromise) {
      svgTextPromise = fetch(src).then((res) => {
        if (!res.ok) throw new Error(`fetch ${src}: ${res.status}`);
        return res.text();
      });
      svgTextPromise.catch(() => {
        svgTextPromise = null;
      });
    }
    return svgTextPromise;
  };

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    if (event.target.closest(".pog-pop-clone")) return;
    const logo = event.target.closest(SELECTOR);
    if (!logo) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    // The logo sits inside the home link: the animation takes over the click.
    event.preventDefault();

    const rect = logo.getBoundingClientRect();
    const host = document.createElement("div");
    host.className = "pog-pop-clone";
    host.setAttribute("aria-hidden", "true");
    Object.assign(host.style, {
      position: "fixed",
      left: `${rect.left}px`,
      top: `${rect.top}px`,
      width: `${rect.width}px`,
      height: `${rect.height}px`,
      margin: "0",
      zIndex: "2000",
      pointerEvents: "none",
      // Grow toward the bottom-right so the top-left corner logo stays
      // fully on screen at max scale.
      transformOrigin: "25% 25%",
    });
    document.body.appendChild(host);

    const play = () => {
      const animation = host.animate(
        [
          { transform: "scale(1)" },
          { transform: `scale(${SCALE})`, offset: PEAK },
          { transform: "scale(1)" },
        ],
        { duration: DURATION, easing: "ease-in-out" }
      );
      for (const eye of host.querySelectorAll(".pog-eye")) {
        eye.style.transformBox = "fill-box";
        eye.style.transformOrigin = "center";
        eye.animate(
          [
            { transform: "scale(1, 1)" },
            { transform: `scale(${EYE_SCALE_X}, ${EYE_SCALE_Y})`, offset: PEAK },
            { transform: "scale(1, 1)" },
          ],
          { duration: DURATION, easing: "ease-in-out" }
        );
      }
      animation.onfinish = () => host.remove();
    };

    loadSvgText(logo.src)
      .then((text) => {
        host.innerHTML = text;
        const svg = host.querySelector("svg");
        if (!svg) throw new Error("no svg root");
        Object.assign(svg.style, { display: "block", width: "100%", height: "100%" });
        play();
      })
      .catch(() => {
        // Fall back to a plain image clone if the SVG cannot be inlined.
        const img = logo.cloneNode();
        Object.assign(img.style, { width: "100%", height: "100%", margin: "0" });
        host.appendChild(img);
        play();
      });
  });
})();
