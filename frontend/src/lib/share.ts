/** Sharing a saved case, benchmark snapshot or policy with the hosted project. */
export type ShareKind = 'case' | 'benchmark' | 'policy'

export interface ShareDestination { configured: boolean; url?: string; project?: string; message?: string; hint?: string | null }
export interface SharePreview {
  transfer: { kind: ShareKind; id: string; revision: string; bytes: number; query_text: boolean }
  summary: { included: string[]; withheld: string[]; question_text: string[]; question_text_count: number }
  destination?: ShareDestination
}
export interface ShareResult extends SharePreview { uploaded: { id: string }; url: string }

/** One line naming where an upload would go, or why it cannot go anywhere yet. */
export function destinationLine(destination: ShareDestination | undefined): string {
  if (!destination) return 'Checking the hosted project…'
  if (!destination.configured) {
    return `Not connected to a hosted project: ${destination.message ?? 'configuration missing'} Set VECTRAUDIT_CLOUD_URL, VECTRAUDIT_CLOUD_TOKEN and VECTRAUDIT_CLOUD_PROJECT before starting the workbench.`
  }
  return `Shares to ${destination.project} at ${destination.url}.`
}

export function sizeLabel(bytes: number): string {
  return bytes < 1024 ? `${bytes} bytes` : `${(bytes / 1024).toFixed(1)} KiB`
}
