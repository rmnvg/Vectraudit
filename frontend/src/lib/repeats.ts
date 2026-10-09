import type { Query } from './workbench'

/** Sparse later traces share the first output, never its unmeasured per-stage timing. */
export function executionView(query: Query | undefined, index: number) {
  const count = query?.repeat_ids?.length || (query?.ids ? 1 : 0)
  const selected = Math.max(0, Math.min(index, Math.max(0, count - 1)))
  const sameOutputs = selected > 0 && query?.repeat_evidence != null && !query.repeat_evidence[String(selected + 1)]
  let evidence = selected === 0 ? query?.evidence : query?.repeat_evidence?.[String(selected + 1)]
  if (sameOutputs) evidence = query?.evidence ? {
    ...query.evidence,
    stages: query.evidence.stages?.map(stage => ({ ...stage, latency_ms: null })),
  } : undefined
  return {
    count, selected, sameOutputs, evidence,
    ids: query?.repeat_ids?.[selected] ?? (selected === 0 ? query?.ids : null),
    metrics: query?.repeat_metrics?.[selected],
  }
}
