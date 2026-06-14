import { cp, mkdir, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = join(root, 'sites', 'marketing');
const build = join(source, 'build');

await rm(build, { recursive: true, force: true });
await mkdir(build, { recursive: true });

for (const entry of ['index.html', 'styles.css', 'script.js', 'static', 'CNAME']) {
  await cp(join(source, entry), join(build, entry), { recursive: true });
}

console.log(`built ${build}`);
