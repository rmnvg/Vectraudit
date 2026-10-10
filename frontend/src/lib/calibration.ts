/** A/A calibration of a saved policy from the workbench (see `vectraudit calibrate`). */
import type { RawPolicy } from './release'

export interface Alarms { comparisons: number; false_alarms: number; by_rule: Record<string, number> }
export interface CalibrationChange { path: string; from: number | null; to: number; observed: number }
export interface CalibrationReport {
  runs: { run_id: string; label: string | null; created_at: string | null }[]
  margin: number
  current: Alarms
  draft: Alarms
  changes: CalibrationChange[]
  draft_policy: RawPolicy | null
  not_noise: string[]
  config_differences: string[]
  notes: string[]
}
export interface CalibrationResult { policy: { id: string | null; name: string; version: number | null }; report: CalibrationReport }
export interface RecordedRun { run_id: string; label?: string; created_at: string; status: string; dataset?: { sha256: string } }

/** `7 false alarms in 20 A/A comparisons (35%)`. */
export function alarmSummary(alarms: Alarms): string {
  const share = alarms.comparisons ? Math.round((100 * alarms.false_alarms) / alarms.comparisons) : 0
  const noun = alarms.false_alarms === 1 ? 'false alarm' : 'false alarms'
  return `${alarms.false_alarms} ${noun} in ${alarms.comparisons} A/A comparisons (${share}%)`
}

/** Complete runs that can be calibrated together: the same benchmark as the first chosen one. */
export function choosableRuns(runs: RecordedRun[], chosen: string[]): RecordedRun[] {
  const complete = runs.filter(r => r.status === 'complete')
  const first = complete.find(r => r.run_id === chosen[0])
  return first?.dataset ? complete.filter(r => r.dataset?.sha256 === first.dataset?.sha256) : complete
}

/** A noise change in words: `metrics.ndcg.max_drop: 0.001 → 0.015 (observed 0.0098)`. */
export function changeLine(change: CalibrationChange): string {
  const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toPrecision(2))
  return `${change.path}: ${change.from === null ? 'not set' : fmt(change.from)} → ${fmt(change.to)} (observed ${fmt(change.observed)})`
}
