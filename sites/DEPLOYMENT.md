# QuarkLM Site Deployment

QuarkLM has two public web surfaces with separate hosting responsibilities:

| Surface | Target | Host | Build command |
| --- | --- | --- | --- |
| Docs | `docs.quark-lm.eidetic-labs.com` | Read the Docs | `npm run docs:build` |
| Marketing | `quark-lm.eidetic-labs.com` | GitHub Pages | `npm run marketing:build` |

The docs site is Docusaurus and follows the Learn, Build, Operate, Secure
structure used by Craik docs. Read the Docs builds it from `.readthedocs.yaml`
and publishes the generated Docusaurus HTML from `sites/docs/build`.

The marketing site must not be Docusaurus. It is a standalone static product
page in the Craik/Stigmem style, built from plain HTML, CSS, and JavaScript by
`scripts/build-marketing.mjs`.

GitHub Pages serves one Pages site per repository, so this repository only uses
Pages for the marketing surface. The docs surface is hosted by Read the Docs so
`docs.quark-lm.eidetic-labs.com` can remain an independent documentation root.

## Read the Docs

1. Import `eidetic-labs/quark-lm` into Read the Docs.
2. Use the repository-root `.readthedocs.yaml` configuration.
3. Configure `docs.quark-lm.eidetic-labs.com` as the canonical custom domain in
   Read the Docs.
4. Point the DNS record for the docs subdomain at the target supplied by Read
   the Docs.

The GitHub Actions `Check QuarkLM Docs` workflow only validates the Docusaurus
build. It does not deploy docs.

## GitHub Pages

The `Deploy QuarkLM Marketing` workflow publishes `sites/marketing/build` to
GitHub Pages. The only Pages `CNAME` in this repository should be
`sites/marketing/CNAME`.

Both surfaces must be reviewed with every promoted release whenever they
reference current product state, eval counts, commands, domains, or roadmap
commitments.
