# jipeng4974.github.io

Personal site of [jipeng4974](https://github.com/jipeng4974), built with
[Hugo](https://gohugo.io/) and the [OINK](https://github.com/pgsty/oink) theme.

## Layout

- `content/` — site content. `writeups/` holds the long-form posts migrated
  from the old terminal-theme site; `tags/` and `search.md` are scaffold pages
  required by the theme; the remaining sections (games, photographs, projects,
  publications, about) are first-level tabs.
- `data/home/` — homepage composition (hero, gallery, cta, footer) per language.
- `static/img/` — images referenced by the posts as
  `https://jipeng4974.github.io/img/<file>`.
- `hugo.yml` — single site configuration file.

## Develop

The theme is expected as a sibling checkout at `../oink` (the `Makefile`
creates a `go.work` replace for it):

```console
$ make b   # build into public/ (dev baseURL, for verification)
$ make d   # hugo server on 127.0.0.1:1313, renders to memory
$ make p   # publish: production build into docs/, ready to commit
```

Deployment is **Deploy from a branch**: Pages is configured to serve from
`main:/docs`. Run `make p`, commit `docs/`, and push to update the live site.
`docs/.nojekyll` is written by `make p` so that `_`-prefixed paths (e.g.
`_print/`) survive GitHub's Jekyll processing.
