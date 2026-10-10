import { describe, expect, it } from 'vitest'
import { destinationLine, sizeLabel } from './share'

describe('share panel text', () => {
  it('names the destination or explains how to connect one', () => {
    expect(destinationLine({ configured: true, url: 'https://va.example.com', project: 'acme/search' })).toBe('Shares to acme/search at https://va.example.com.')
    const missing = destinationLine({ configured: false, message: 'VECTRAUDIT_CLOUD_URL is not set.' })
    expect(missing).toMatch(/^Not connected to a hosted project: VECTRAUDIT_CLOUD_URL is not set\./)
    expect(missing).toMatch(/VECTRAUDIT_CLOUD_PROJECT before starting the workbench\.$/)
    expect(destinationLine(undefined)).toBe('Checking the hosted project…')
  })
  it('shows transfer size readably', () => {
    expect(sizeLabel(512)).toBe('512 bytes')
    expect(sizeLabel(2048)).toBe('2.0 KiB')
  })
})
