to build this project, make b or make p, see Makefile
the theme repo is available as a sibling ../oink, only edit oink theme when it's necessary. 
source ~/.zshrc to add hugo and go to $PATH
photos section pages use progressive image loading: layouts/_markup/render-image.html wraps all but the first image of each photos page in placeholder frames, assets/js/photo-gallery.js (injected via layouts/_partials/hooks/body-end.html) fetches them on scroll with a concurrency cap, streamed download progress and retries; styles live in assets/scss/_styles_project.scss