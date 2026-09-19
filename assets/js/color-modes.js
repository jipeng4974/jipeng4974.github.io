// Color modes for the shell: light → classic (古韵) → dark → …
//
// The theme's dark-mode.js owns the light/dark primitives and the prepaint
// script still restores the stored value from 'td-color-theme'. This script
// widens that vocabulary: it intercepts clicks on the theme toggle in the
// capture phase (so the theme's binary light/dark listener never sees them),
// cycles through the full list, and stores the selected value back under the
// same key. Adding a future theme (e.g. highcontrast) means adding it to
// THEMES, NAMES and a `[data-bs-theme='<name>']` style block.
(() => {
  const KEY = "td-color-theme";
  const THEMES = ["light", "classic", "dark"];
  const NAMES = {
    light: "light",
    classic: "classic",
    dark: "dark",
  };

  const root = document.documentElement;
  const getStored = () => {
    try {
      return localStorage.getItem(KEY);
    } catch (_) {
      return null;
    }
  };

  const setStored = (theme) => {
    try {
      localStorage.setItem(KEY, theme);
    } catch (_) {
      // A blocked storage policy must not disable the in-page control.
    }
  };

  const current = () => {
    const value = root.getAttribute("data-bs-theme");
    return THEMES.includes(value) ? value : "light";
  };

  const next = () => THEMES[(THEMES.indexOf(current()) + 1) % THEMES.length];

  // Hover/assistive text keeps the theme's original description and appends
  // the *next* mode name on the right, e.g. "Toggle theme · classic".
  const syncToggleLabels = () => {
    const nextName = NAMES[next()] || next();
    document.querySelectorAll("[data-td-theme-toggle]").forEach((toggle) => {
      if (!toggle.dataset.tdThemeBaseTitle) {
        toggle.dataset.tdThemeBaseTitle = toggle.getAttribute("title") || "";
      }
      if (!toggle.dataset.tdThemeBaseLabel) {
        toggle.dataset.tdThemeBaseLabel = toggle.getAttribute("aria-label") || "";
      }

      const title = toggle.dataset.tdThemeBaseTitle;
      const label = toggle.dataset.tdThemeBaseLabel;
      toggle.setAttribute("title", title ? `${title} · ${nextName}` : nextName);
      toggle.setAttribute(
        "aria-label",
        label ? `${label} · ${nextName}` : nextName
      );
    });
  };

  const apply = (theme) => {
    root.setAttribute("data-bs-theme", theme);
    setStored(theme);
    syncToggleLabels();
  };

  const cycle = () => apply(next());

  // Stop the theme's own binary toggle before it reaches the button; the
  // capture phase is the only place where stopPropagation can suppress a
  // listener registered directly on the target.
  document.addEventListener(
    "click",
    (event) => {
      if (!(event.target instanceof Element)) return;
      if (!event.target.closest("[data-td-theme-toggle]")) return;
      event.preventDefault();
      event.stopPropagation();
      cycle();
    },
    true
  );

  // If the system colour preference flips while classic mode is selected, the
  // theme listener would overwrite it with light/dark; restore the explicit
  // choice afterwards.
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => {
      if (getStored() === "classic") apply("classic");
    });

  syncToggleLabels();
})();
