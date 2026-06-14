# QuarkLM Site Deployment

QuarkLM has two public web surfaces:

| Surface | Target | Build command |
| --- | --- | --- |
| Docs | `docs.quark-lm.eidetic-labs.com` | `npm run docs:build` |
| Marketing | `quark-lm.eidetic-labs.com` | `npm run marketing:build` |

The docs site is Docusaurus and follows the Learn, Build, Operate, Secure
structure used by Craik docs.

The marketing site must not be Docusaurus. It is a standalone static product
page in the Craik/Stigmem style, built from plain HTML, CSS, and JavaScript by
`scripts/build-marketing.mjs`.

GitHub Pages serves one Pages site per repository. If both custom domains need
to be independent roots, deploy these artifacts from separate Pages repos or add
DNS/proxy routing that maps each domain to the correct artifact. The included
workflows are split by site so either artifact can be deployed independently.

Both surfaces must be reviewed with every promoted release whenever they
reference current product state, eval counts, commands, domains, or roadmap
commitments.
