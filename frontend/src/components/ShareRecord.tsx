import { useEffect, useState } from 'react'
import { guidedRequest } from '../lib/guided'
import { destinationLine, sizeLabel } from '../lib/share'
import type { SharePreview, ShareKind, ShareResult } from '../lib/share'

const button = 'rounded-lg border border-line px-3 py-2 text-sm hover:bg-surface-2 disabled:opacity-40'

/** Preview exactly what would leave this machine, then share it only on explicit confirmation. */
export function ShareRecord({ token, kind, id, label }: { token: string; kind: ShareKind; id: string; label: string }) {
  const [open, setOpen] = useState(false)
  const [withText, setWithText] = useState(false)
  const [loaded, setLoaded] = useState<{ key: string; preview: SharePreview } | null>(null)
  const [result, setResult] = useState<ShareResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const key = `${kind}:${id}:${withText}`
  const preview = loaded?.key === key ? loaded.preview : null

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    guidedRequest<SharePreview>('share/preview', token, { kind, id, share_query_text: withText }, controller.signal)
      .then(next => setLoaded({ key: `${kind}:${id}:${withText}`, preview: next }))
      .catch(e => { if (!controller.signal.aborted) setError(String(e instanceof Error ? e.message : e)) })
    return () => controller.abort()
  }, [open, withText, token, kind, id])

  async function share() {
    setBusy(true)
    setError('')
    try {
      setResult(await guidedRequest<ShareResult>('share', token, { kind, id, share_query_text: withText, confirm: true }))
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e))
    } finally {
      setBusy(false)
    }
  }

  if (!open) return <button className={button} onClick={() => { setResult(null); setError(''); setOpen(true) }}>Share with hosted project…</button>
  const destination = preview?.destination
  return <section aria-label={`Share ${label}`} className="space-y-2 rounded-lg border border-line p-3 text-sm">
    <div className="flex items-start justify-between gap-3"><h4 className="font-semibold">Share {label}</h4><button className={button} onClick={() => setOpen(false)}>Close</button></div>
    <p className={destination && !destination.configured ? 'text-amber-600' : 'text-muted'}>{destinationLine(destination)}</p>
    {preview && <>
      <p className="font-medium">Sent</p>
      <ul className="list-disc pl-5">{preview.summary.included.map(item => <li key={item}>{item}</li>)}</ul>
      <p className="font-medium">Stays on this machine</p>
      <ul className="list-disc pl-5 text-muted">{preview.summary.withheld.map(item => <li key={item}>{item}</li>)}</ul>
      {kind !== 'policy' && <label className="flex items-start gap-2"><input type="checkbox" checked={withText} onChange={e => { setError(''); setResult(null); setWithText(e.target.checked) }} /><span>Also share question text{kind === 'case' ? ' and document titles' : ''}<span className="block text-xs text-muted">Off by default. Only text verified against the saved benchmark is sent; the hosted page marks the record as including text.</span></span></label>}
      {withText && <p className="text-xs text-muted">{preview.summary.question_text_count ? `${preview.summary.question_text_count} question text(s) will be sent${preview.summary.question_text.length ? `, for example: “${preview.summary.question_text[0]}”` : ''}.` : 'No verified question text is available, so none will be sent.'}</p>}
      <p className="text-xs text-muted">Revision {preview.transfer.revision.slice(0, 8)} · {sizeLabel(preview.transfer.bytes)}. Shared revisions cannot be changed; a later local revision uploads as a new record.</p>
      <button className={button} disabled={busy || !destination?.configured} onClick={() => void share()}>{busy ? 'Sharing…' : 'Share now'}</button>
    </>}
    {result && <p role="status">Shared. <a className="text-teal-600 underline" href={result.url} target="_blank" rel="noreferrer">Open it in the hosted project</a></p>}
    {error && <p role="alert" className="text-red-600">{error}</p>}
  </section>
}
