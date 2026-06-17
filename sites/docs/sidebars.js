// @ts-check
//
// QuarkLM docs sidebars — restructured for task-first wayfinding, modeled on
// the Craik page-craft layout.
//
// Top-level structure: Learn · Build · Operate · Secure.
// Each sidebar starts with its own landing page and then organizes the rest
// of its docs into numbered, task-focused categories. Every existing doc id
// is preserved — none are renamed, moved, or dropped.
//

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // ─────────────────────────────────────────────────────────────────────
  // LEARN — Understand what QuarkLM is and the research behind it
  // ─────────────────────────────────────────────────────────────────────
  learn: [
    'learn/index',
    {
      type: 'category',
      label: '1 · Getting oriented',
      collapsed: false,
      items: [
        'learn/project-overview',
        'learn/language-model',
        'learn/self-improvement-loop',
      ],
    },
    {
      type: 'category',
      label: '2 · Research grounding',
      collapsed: false,
      items: [
        'learn/research-grounding',
        'learn/research-implementation-map',
        'learn/forward-research-plan',
        'learn/deep-research-review',
      ],
    },
    {
      type: 'category',
      label: '3 · Evidence & audit',
      collapsed: true,
      items: [
        'learn/current-evidence',
        'learn/historical-evidence',
        'learn/branch-diversity-research',
        'learn/open-source-mechanics-audit',
      ],
    },
  ],

  // ─────────────────────────────────────────────────────────────────────
  // BUILD — Install QuarkLM and put the transformer to work
  // ─────────────────────────────────────────────────────────────────────
  build: [
    'build/index',
    {
      type: 'category',
      label: '1 · Getting started',
      collapsed: false,
      items: [
        'build/quickstart',
        'build/admission-workflow',
      ],
    },
    {
      type: 'category',
      label: '2 · The transformer',
      collapsed: false,
      items: [
        'build/transformer',
        'build/transformer-responsibilities',
        'build/transformer-screen-history',
        'build/generated-probes',
      ],
    },
  ],

  // ─────────────────────────────────────────────────────────────────────
  // OPERATE — Run releases, keep the corpus clean, prove what shipped
  // ─────────────────────────────────────────────────────────────────────
  operate: [
    'operate/index',
    {
      type: 'category',
      label: '1 · Releasing',
      collapsed: false,
      items: [
        'operate/release-candidate',
        'operate/release-discipline',
      ],
    },
    {
      type: 'category',
      label: '2 · Corpus & experiments',
      collapsed: false,
      items: [
        'operate/experiment-registry',
        'operate/corpus-hygiene',
        'operate/candidate-quarantine',
        'operate/training-recipes',
      ],
    },
    {
      type: 'category',
      label: '3 · Verification & provenance',
      collapsed: true,
      items: [
        'operate/closed-world-verifier',
        'operate/provenance',
        'operate/docs-drift',
      ],
    },
  ],

  // ─────────────────────────────────────────────────────────────────────
  // SECURE — Hold the boundaries that keep the model trustworthy
  // ─────────────────────────────────────────────────────────────────────
  secure: [
    'secure/index',
    {
      type: 'category',
      label: '1 · Boundaries & policy',
      collapsed: false,
      items: [
        'secure/purity-boundary',
        'secure/prompt-leakage',
        'secure/unknown-policy',
      ],
    },
  ],
};

module.exports = sidebars;
