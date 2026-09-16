// Pog pop: clicking the brand logo plays a grow-then-shrink overlay
// animation on a fixed-position clone, so the layout never shifts.
(() => {
  const SELECTOR = ".td-shell-sidebar__brand img, .td-shell-subnav__brand img";
  const SCALE = 6;
  const GROW_MS = 500;
  const SHRINK_MS = 900;

  document.addEventListener("click", (event) => {
    if (event.target instanceof Element && event.target.closest(".pog-pop-clone")) return;
    const logo = event.target instanceof Element ? event.target.closest(SELECTOR) : null;
    if (!logo) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    // The logo sits inside the home link: the animation takes over the click.
    event.preventDefault();

    const rect = logo.getBoundingClientRect();
    const clone = logo.cloneNode();
    clone.className = "pog-pop-clone";
    clone.setAttribute("aria-hidden", "true");
    Object.assign(clone.style, {
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
      transformOrigin: "35% 35%",
    });
    document.body.appendChild(clone);

    const animation = clone.animate(
      [
        { transform: "scale(1)" },
        { transform: `scale(${SCALE})`, offset: GROW_MS / (GROW_MS + SHRINK_MS) },
        { transform: "scale(1)" },
      ],
      { duration: GROW_MS + SHRINK_MS, easing: "ease-in-out" }
    );
    animation.onfinish = () => clone.remove();
  });
})();
