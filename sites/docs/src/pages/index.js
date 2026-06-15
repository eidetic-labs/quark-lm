import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import state from '../../../shared/current-state.json';
import './index.css';

const sections = [
  {
    number: '01',
    label: 'Learn',
    title: 'Concepts and product model',
    body: 'Understand closed-world learning, the language model, the admitted dataset, and the current evidence.',
    to: '/docs/learn/',
    chips: ['vision', 'model', 'self-improvement', 'evidence'],
  },
  {
    number: '02',
    label: 'Build',
    title: 'Run and extend QuarkLM',
    body: 'Generate curriculum, train random weights, admit new facts, and add generated probes without crossing the purity boundary.',
    to: '/docs/build/',
    chips: ['quickstart', 'admission', 'probes', 'commands'],
  },
  {
    number: '03',
    label: 'Operate',
    title: 'Promote releases with evidence',
    body: 'Use self-improvement reports, self-diagnosis, forgetting audits, provenance snapshots, and docs freshness gates.',
    to: '/docs/operate/',
    chips: ['release gates', 'reports', 'provenance', 'docs drift'],
  },
  {
    number: '04',
    label: 'Secure',
    title: 'Keep the world closed',
    body: 'Guard against pretrained weights, unledgered text, prompt leakage, and claims outside the corpus.',
    to: '/docs/secure/',
    chips: ['purity', 'leakage', 'unknowns', 'boundaries'],
  },
];

const paths = [
  {
    audience: 'New to QuarkLM',
    title: 'Read the model, then run a smoke cycle',
    steps: [
      ['Language model', '/docs/learn/language-model/'],
      ['Quickstart', '/docs/build/quickstart/'],
      ['Current evidence', '/docs/learn/current-evidence/'],
    ],
  },
  {
    audience: 'Teaching a new fact',
    title: 'Admit memory, generate probes, retrain weights',
    steps: [
      ['Admission workflow', '/docs/build/admission-workflow/'],
      ['Generated probes', '/docs/build/generated-probes/'],
      ['Release discipline', '/docs/operate/release-discipline/'],
    ],
  },
  {
    audience: 'Protecting the experiment',
    title: 'Audit provenance, leakage, and docs freshness',
    steps: [
      ['Purity boundary', '/docs/secure/purity-boundary/'],
      ['Prompt leakage', '/docs/secure/prompt-leakage/'],
      ['Docs drift', '/docs/operate/docs-drift/'],
    ],
  },
];

const primitives = [
  ['corpus.ledger', 'Ledger', 'The explicit list of files allowed to influence training or evaluation.'],
  ['admission.log', 'Admitted memory', 'Structured facts that become learnable only after admission.'],
  ['probe.audit', 'Generated probes', 'Direct and paraphrase checks derived from the admitted-memory log.'],
  ['weight.run', 'Versioned weights', 'Randomly initialized checkpoints promoted only with recorded metrics.'],
  ['forgetting.audit', 'Forgetting audit', 'A comparison against the previous promoted report.'],
  ['diagnosis.report', 'Self-diagnosis', 'Rule-based repair recommendations derived from the run report, with no external model.'],
  ['verifier.report', 'Closed-world verifier', 'Deterministic approval for candidate checks and training plans, with no external model.'],
  ['recipe.run', 'Training recipe', 'A reproducible record of model, tokenizer, data, objective, optimizer, artifacts, gates, and rerun details.'],
  ['docs.release', 'Docs gate', 'README, docs, and marketing content updated with each release when they reference current state.'],
];

function SectionCard({ item }) {
  return (
    <li>
      <Link className="qdocs-nav-card" to={item.to}>
        <div className="qdocs-nav-card__head">
          <span>{item.number}</span>
          <strong>{item.label}</strong>
        </div>
        <h3>{item.title}</h3>
        <p>{item.body}</p>
        <ul aria-label={`${item.label} topics`}>
          {item.chips.map((chip) => (
            <li key={chip}>{chip}</li>
          ))}
        </ul>
        <span className="qdocs-card-cta">Open {item.label}</span>
      </Link>
    </li>
  );
}

export default function Home() {
  return (
    <Layout
      title="QuarkLM Documentation"
      description="Documentation for QuarkLM, a closed-world language model prototype from Eidetic Labs."
    >
      <main className="qdocs-home">
        <section className="qdocs-hero" aria-labelledby="qdocs-title">
          <div className="qdocs-hero__grid" aria-hidden="true" />
          <div className="qdocs-hero__copy">
            <p className="qdocs-eyebrow">
              <span />
              Documentation · {state.currentVersion} · Research prototype
            </p>
            <h1 id="qdocs-title">
              <span>Big idea.</span>
              <span className="qdocs-accent">Tiny package.</span>
            </h1>
            <p className="qdocs-lede">
              QuarkLM is a closed-world language model: random weights, no pretrained
              tokenizer, no external embeddings, and learning only through the admitted corpus.
              These docs cover the model, build loop, operating discipline, and security boundary.
            </p>
            <div className="qdocs-actions">
              <Link className="qdocs-button qdocs-button--primary" to="/docs/build/quickstart/">
                Quickstart
              </Link>
              <Link className="qdocs-button qdocs-button--ghost" to="/docs/learn/current-evidence/">
                Read the evidence
              </Link>
              <code className="qdocs-snippet">$ python3 -m {state.internalImportPath}.self_improve</code>
            </div>
            <dl className="qdocs-stats">
              <div>
                <dt>Run</dt>
                <dd>{state.currentRun}</dd>
              </div>
              <div>
                <dt>Admission probes</dt>
                <dd>{state.directAdmissionProbes} direct · {state.admissionParaphraseProbes} paraphrase · {state.glossaryProbes} glossary</dd>
              </div>
              <div>
                <dt>Boundary</dt>
                <dd>no pretrained weights · no external embeddings</dd>
              </div>
              <div>
                <dt>Diagnosis</dt>
                <dd>{state.selfDiagnosis} · no external model</dd>
              </div>
            </dl>
          </div>
          <aside className="qdocs-stage" aria-label="QuarkLM release state">
            <div className="qdocs-stack">
              <div className="qdocs-stack__chrome">
                <span />
                <span />
                <span />
                <code>~/quark-lm · release</code>
              </div>
              <ol>
                <li><span>L6</span><strong>Docs gate</strong><code>README · Docusaurus · marketing</code></li>
                <li><span>L5</span><strong>Self-diagnosis</strong><code>{state.selfDiagnosis} · {state.selfDiagnosisBlockers} blockers</code></li>
                <li><span>L4</span><strong>Forgetting</strong><code>{state.forgettingAudit}</code></li>
                <li><span>L3</span><strong>Probe audit</strong><code>{state.directAdmissionProbes} + {state.admissionParaphraseProbes} + {state.glossaryProbes}</code></li>
                <li><span>L2</span><strong>Weight update</strong><code>random init · versioned checkpoint</code></li>
                <li><span>L1</span><strong>Admission</strong><code>{state.admittedFacts} facts · ledgered</code></li>
                <li><span>L0</span><strong>Corpus</strong><code>glossary · grammar · admissions</code></li>
              </ol>
            </div>
          </aside>
        </section>

        <section className="qdocs-section" aria-labelledby="qdocs-nav-title">
          <header className="qdocs-section-head">
            <p className="qdocs-eyebrow"><span /> Navigate</p>
            <h2 id="qdocs-nav-title">Four entry points into QuarkLM.</h2>
          </header>
          <ol className="qdocs-nav-grid">
            {sections.map((item) => <SectionCard key={item.label} item={item} />)}
          </ol>
        </section>

        <section className="qdocs-section qdocs-section--split" aria-labelledby="qdocs-path-title">
          <header className="qdocs-section-head">
            <p className="qdocs-eyebrow"><span /> Pick a path</p>
            <h2 id="qdocs-path-title">Where are you trying to go?</h2>
            <p>
              Curated paths for the most common moves in the prototype: understand the
              experiment, admit new memory, and promote evidence without drift.
            </p>
          </header>
          <ol className="qdocs-path-grid">
            {paths.map((path, index) => (
              <li className="qdocs-path" key={path.audience}>
                <p><span>{String(index + 1).padStart(2, '0')}</span>{path.audience}</p>
                <h3>{path.title}</h3>
                <ol>
                  {path.steps.map(([label, to], stepIndex) => (
                    <li key={label}>
                      <Link to={to}>
                        <span>{String(stepIndex + 1).padStart(2, '0')}</span>
                        {label}
                      </Link>
                    </li>
                  ))}
                </ol>
              </li>
            ))}
          </ol>
        </section>

        <section className="qdocs-section" aria-labelledby="qdocs-primitive-title">
          <header className="qdocs-section-head">
            <p className="qdocs-eyebrow"><span /> Primitives</p>
            <h2 id="qdocs-primitive-title">The loop is seven auditable objects.</h2>
          </header>
          <ul className="qdocs-primitive-grid">
            {primitives.map(([type, title, body]) => (
              <li key={type} className={clsx('qdocs-primitive', type === 'docs.release' && 'qdocs-primitive--accent')}>
                <p>{type}</p>
                <h3>{title}</h3>
                <span>{body}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="qdocs-band" aria-labelledby="qdocs-band-title">
          <div>
            <p className="qdocs-eyebrow"><span /> Eidetic Labs</p>
            <h2 id="qdocs-band-title">Need the product story or the source?</h2>
            <p>
              The marketing page carries the concise product position. The repository and
              these docs remain the source of truth for commands, evidence, and release gates.
            </p>
          </div>
          <div className="qdocs-band__actions">
            <a className="qdocs-button qdocs-button--primary" href={state.marketingUrl}>Product site</a>
            <a className="qdocs-button qdocs-button--ghost" href={state.githubUrl}>GitHub · eidetic-labs/quark-lm</a>
          </div>
        </section>
      </main>
    </Layout>
  );
}
