/* Render every view in both themes and report anything the browser complains
   about. Screenshots catch layout problems the palette validator cannot see:
   clipped labels, overflow, collisions. */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const BASE = process.env.BASE || 'http://127.0.0.1:5001'
const OUT = 'shots'
mkdirSync(OUT, { recursive: true })

// Drive the Chrome already installed on this machine rather than downloading
// a private Chromium build.
const browser = await chromium.launch({ channel: 'chrome' })
const problems = []

for (const theme of ['dark', 'light']) {
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } })
  page.on('console', (m) => {
    if (m.type() === 'error') problems.push(`[${theme}] console: ${m.text()}`)
  })
  page.on('pageerror', (e) => problems.push(`[${theme}] pageerror: ${e.message}`))
  page.on('requestfailed', (r) =>
    problems.push(`[${theme}] request failed: ${r.url()} ${r.failure()?.errorText}`)
  )

  for (const name of ['lab', 'blunders', 'coaches', 'method']) {
    await page.goto(`${BASE}/?theme=${theme}#${name}`, { waitUntil: 'networkidle' })
    await page.waitForTimeout(1600)
    await page.screenshot({ path: `${OUT}/${name}-${theme}.png`, fullPage: true })
  }

  // Does the page ever scroll sideways? A horizontal body scrollbar is a bug.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth
  )
  if (overflow > 2) problems.push(`[${theme}] horizontal overflow: ${overflow}px`)

  await page.close()
}

// Narrow viewport: the layout has to survive a laptop half-screen.
const narrow = await browser.newPage({ viewport: { width: 760, height: 1000 } })
await narrow.goto(BASE, { waitUntil: 'networkidle' })
await narrow.waitForTimeout(1200)
await narrow.screenshot({ path: `${OUT}/lab-narrow.png`, fullPage: true })
const nOver = await narrow.evaluate(
  () => document.documentElement.scrollWidth - document.documentElement.clientWidth
)
if (nOver > 2) problems.push(`[narrow] horizontal overflow: ${nOver}px`)
await narrow.close()

await browser.close()

console.log(problems.length ? 'PROBLEMS:\n' + problems.join('\n') : 'no console errors, no overflow')
