const path = require('path');

const readTheDocsBaseUrl = () => {
  if (process.env.DOCUSAURUS_BASE_URL) {
    return process.env.DOCUSAURUS_BASE_URL;
  }

  if (process.env.READTHEDOCS === 'True') {
    const language = process.env.READTHEDOCS_LANGUAGE || 'en';
    const version = process.env.READTHEDOCS_VERSION || 'latest';
    return `/${language}/${version}/`;
  }

  return '/';
};

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'QuarkLM',
  tagline: 'Big idea. Tiny package.',
  favicon: 'img/brand/favicon.svg',
  url: 'https://docs.quark-lm.eidetic-labs.com',
  baseUrl: readTheDocsBaseUrl(),
  organizationName: 'eidetic-labs',
  projectName: 'quark-lm',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  customFields: {
    statePath: path.resolve(__dirname, '../shared/current-state.json'),
  },
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: 'docs',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    image: 'img/brand/og-image.svg',
    navbar: {
      title: 'QuarkLM',
      logo: {
        alt: 'QuarkLM docs',
        src: 'img/brand/quarklm_mark.svg',
        srcDark: 'img/brand/quarklm_mark_inverse.svg',
      },
      items: [
        { to: '/docs/learn/', label: 'Learn', position: 'left' },
        { to: '/docs/build/', label: 'Build', position: 'left' },
        { to: '/docs/operate/', label: 'Operate', position: 'left' },
        { to: '/docs/secure/', label: 'Secure', position: 'left' },
        {
          href: 'https://quark-lm.eidetic-labs.com',
          label: 'quark-lm.eidetic-labs.com',
          position: 'right',
        },
        {
          href: 'https://github.com/eidetic-labs/quark-lm',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Learn',
          items: [
            { label: 'Overview', to: '/docs/learn/' },
            { label: 'Language model', to: '/docs/learn/language-model/' },
            { label: 'Evidence', to: '/docs/learn/current-evidence/' },
          ],
        },
        {
          title: 'Build',
          items: [
            { label: 'Overview', to: '/docs/build/' },
            { label: 'Quickstart', to: '/docs/build/quickstart/' },
            { label: 'Admission workflow', to: '/docs/build/admission-workflow/' },
          ],
        },
        {
          title: 'Operate',
          items: [
            { label: 'Overview', to: '/docs/operate/' },
            { label: 'Release discipline', to: '/docs/operate/release-discipline/' },
            { label: 'Provenance', to: '/docs/operate/provenance/' },
          ],
        },
        {
          title: 'Secure',
          items: [
            { label: 'Overview', to: '/docs/secure/' },
            { label: 'Purity boundary', to: '/docs/secure/purity-boundary/' },
            { label: 'Prompt leakage', to: '/docs/secure/prompt-leakage/' },
          ],
        },
        {
          title: 'Project',
          items: [
            { label: 'Marketing site', href: 'https://quark-lm.eidetic-labs.com' },
            { label: 'GitHub', href: 'https://github.com/eidetic-labs/quark-lm' },
            { label: 'Eidetic Labs', href: 'https://eidetic-labs.com' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Eidetic Labs. QuarkLM research prototype.`,
    },
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
  },
};

module.exports = config;
