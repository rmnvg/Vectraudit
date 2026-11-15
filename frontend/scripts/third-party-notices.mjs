// Write the licence notices of every npm package bundled into the UI (runtime dependencies,
// not dev tooling) to src/vectraudit/THIRD_PARTY_NOTICES.txt, which ships in the wheel.
// The minified bundle drops licence comments, and these licences require the notices to
// travel with the product. Run after `npm ci`; `--check` fails on drift (CI).
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), '..')
const DEST = join(FRONTEND, '../src/vectraudit/THIRD_PARTY_NOTICES.txt')
const LICENCE_FILE = /^(licen[cs]e|copying|notice)(\.(md|txt|markdown))?$/i

function licenceFiles(dir, depth = 0) {
  const entries = readdirSync(dir).sort()
  const own = entries.filter((e) => LICENCE_FILE.test(e)).map((e) => join(dir, e))
  if (own.length || depth >= 2) return own
  // Packages that vendor others (victory-vendor bundles d3) keep the licences one level down.
  return entries
    .filter((e) => e !== 'node_modules' && statSync(join(dir, e)).isDirectory())
    .flatMap((e) => licenceFiles(join(dir, e), depth + 1))
}

// Packages that are installed only because a required peer depends on them and that no emitted
// chunk contains (checked by tests/test_third_party_notices.py). Their licence is the one their
// package.json declares; they ship no licence file to copy.
const NOT_BUNDLED = new Set(['@duckdb/duckdb-wasm'])

// Bundled packages that ship no licence file: the licence text of their upstream repository,
// copied verbatim into scripts/licences/ (their package.json declares the same licence).
const VENDORED = { 'embedding-atlas': 'licences/embedding-atlas.LICENSE' }

const lock = JSON.parse(readFileSync(join(FRONTEND, 'package-lock.json'), 'utf8'))
const sections = []
for (const [path, pkg] of Object.entries(lock.packages).sort(([a], [b]) => a.localeCompare(b))) {
  if (!path.startsWith('node_modules/') || pkg.dev || pkg.devOptional) continue
  const name = path.slice(path.lastIndexOf('node_modules/') + 'node_modules/'.length)
  const dir = join(FRONTEND, path)
  if (!existsSync(dir)) throw new Error(`${name} is not installed; run npm ci first`)
  const files = licenceFiles(dir)
  if (!files.length && VENDORED[name]) files.push(join(import.meta.dirname, VENDORED[name]))
  if (!files.length && NOT_BUNDLED.has(name)) {
    sections.push(`${name}@${pkg.version} (${pkg.license ?? 'see its package.json'})\n\nNot bundled into the web UI: installed as a dependency of a required peer and tree-shaken out of every chunk. It ships no licence file.`)
    continue
  }
  if (!files.length) throw new Error(`${name}@${pkg.version} has no licence file`)
  const texts = files.map((f) =>
    files.length > 1 ? `[${relative(dir, f)}]\n${readFileSync(f, 'utf8').trim()}` : readFileSync(f, 'utf8').trim(),
  )
  sections.push(`${name}@${pkg.version} (${pkg.license ?? 'see below'})\n\n${texts.join('\n\n')}`)
}

const rule = '\n\n' + '-'.repeat(80) + '\n\n'
const text =
  'Third-party software bundled in the Vectraudit web UI\n\n' +
  'The Vectraudit UI includes the following open-source packages, each under its own licence.\n' +
  'Python dependencies are installed separately and carry their own licence metadata.' +
  rule +
  sections.join(rule) +
  '\n'
// Trailing whitespace and CRLF in upstream licence files would trip the pre-commit hooks.
const out = text.replace(/\r\n?/g, '\n').replace(/[ \t]+$/gm, '')

if (process.argv.includes('--check')) {
  if (!existsSync(DEST) || readFileSync(DEST, 'utf8') !== out) {
    console.error('THIRD_PARTY_NOTICES.txt is out of date; run `node scripts/third-party-notices.mjs`.')
    process.exit(1)
  }
} else {
  writeFileSync(DEST, out)
  console.log(`Wrote ${sections.length} notices to ${relative(process.cwd(), DEST)}`)
}
