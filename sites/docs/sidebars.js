/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  learn: [
    {
      type: 'category',
      label: 'Learn',
      link: { type: 'doc', id: 'learn/index' },
      items: [
        'learn/language-model',
        'learn/self-improvement-loop',
        'learn/research-grounding',
        'learn/open-source-mechanics-audit',
        'learn/forward-research-plan',
        'learn/deep-research-review',
        'learn/research-implementation-map',
        'learn/current-evidence',
      ],
    },
  ],
  build: [
    {
      type: 'category',
      label: 'Build',
      link: { type: 'doc', id: 'build/index' },
      items: [
        'build/quickstart',
        'build/admission-workflow',
        'build/generated-probes',
        'build/transformer',
      ],
    },
  ],
  operate: [
    {
      type: 'category',
      label: 'Operate',
      link: { type: 'doc', id: 'operate/index' },
      items: [
        'operate/release-discipline',
        'operate/experiment-registry',
        'operate/corpus-hygiene',
        'operate/candidate-quarantine',
        'operate/provenance',
        'operate/docs-drift',
      ],
    },
  ],
  secure: [
    {
      type: 'category',
      label: 'Secure',
      link: { type: 'doc', id: 'secure/index' },
      items: [
        'secure/purity-boundary',
        'secure/prompt-leakage',
        'secure/unknown-policy',
      ],
    },
  ],
};

module.exports = sidebars;
