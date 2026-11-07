import { expect, test } from '@playwright/test'

const ATLAS_URL = 'http://127.0.0.1:8796'

// The atlas draws with WebGL 2; headless Chrome needs the GPU flags (see CLAUDE.md).
test.use({ launchOptions: { args: ['--enable-gpu', '--ignore-gpu-blocklist'] } })

test('atlas: loads quickly, finds the uncovered group and locates failing questions', async ({ page }) => {
  await page.setViewportSize({ width: 1300, height: 1400 }) // the map is 800px tall
  await page.goto(`${ATLAS_URL}/#/workbench`)
  const started = Date.now()
  await page.getByLabel('Open local artifact').selectOption('atlas-e2e.json')
  await expect(page.getByTestId('corpus-atlas')).toBeVisible()
  await expect(page.locator('[data-testid=corpus-atlas] canvas').first()).toBeVisible({ timeout: 10_000 })
  expect(Date.now() - started).toBeLessThan(3_000)

  // The finding is a hypothesis, names the group, and the region offers documents to judge.
  const finding = page.getByLabel('Coverage findings')
  await expect(finding).toContainText('of documents form a group no benchmark question is close to')
  await expect(finding).toContainText('refund')
  await expect(finding).toContainText('(hypothesis)')
  await expect(page.getByTestId('region-uncovered')).toHaveCount(1)
  await expect(page.getByRole('button', { name: /documents to judge/ })).toBeVisible()

  // Failing questions are drawn and counted (one of the 14 "failing" questions is relevant to
  // the very document it was given, so it passes); held-out questions are not drawn at all.
  const legend = page.getByLabel('Legend')
  await expect(legend).toContainText('Question: failed (13)')
  await expect(legend).toContainText('Question: passed (43)')
  await expect(page.getByTestId('corpus-atlas')).toContainText('Held-out questions are not drawn.')
  await expect(page.getByTestId('corpus-atlas')).not.toContainText('heldout-1')
  await expect(page.getByTestId('corpus-atlas')).toContainText('distorts distance')

  // A rectangle selection over the map (the rectangle tool) lists what is there in the side panel,
  // with the right action for each kind of point.
  const canvas = page.locator('[data-testid=corpus-atlas] canvas').first()
  await canvas.scrollIntoViewIfNeeded()
  const box = (await canvas.boundingBox())!
  const panel = page.getByLabel('Selection')
  await page.getByTitle(/Toggle rectangle selection mode/).click()
  await page.mouse.move(box.x + 20, box.y + 20)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 6 })
  await page.mouse.move(box.x + box.width - 20, box.y + box.height - 60, { steps: 6 })
  await page.mouse.up()
  await expect(panel.locator('.font-mono').first()).toBeVisible()
  await expect(panel).toContainText('more not listed')
  await expect(panel.getByRole('button', { name: /Grade in the Guided workflow|Investigate in Pipeline runs/ }).first()).toBeVisible()
})
