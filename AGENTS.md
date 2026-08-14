to build this project, make b or make p, see Makefile
the theme repo is available as a sibling ../oink, only edit oink theme when it's necessary. 
source ~/.zshrc to add hugo and go to $PATH
photos section pages use progressive image loading: layouts/_markup/render-image.html wraps every image of each photos page in placeholder frames, assets/js/photo-gallery.js (injected via layouts/_partials/hooks/body-end.html) fetches them on scroll with a concurrency cap, streamed download progress and retries, and adds a fullscreen lightbox (click to open, prev/next arrows, ESC/X/backdrop to close, live progress for pending photos); styles live in assets/scss/_styles_project.scss

In writeups, English version has no suffix, Chinese version has suffix .zh.md.