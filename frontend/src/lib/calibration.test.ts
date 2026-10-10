import { describe, expect, it } from 'vitest'
import { alarmSummary, changeLine, choosableRuns } from './calibration'

describe('calibration panel', () => {
  it('summarizes false alarms with their denominator', () => {
    expect(alarmSummary({ comparisons: 20, false_alarms: 7, by_rule: {} })).toBe('7 false alarms in 20 A/A comparisons (35%)')
    expect(alarmSummary({ comparisons: 20, false_alarms: 1, by_rule: {} })).toBe('1 false alarm in 20 A/A comparisons (5%)')
    expect(alarmSummary({ comparisons: 0, false_alarms: 0, by_rule: {} })).toBe('0 false alarms in 0 A/A comparisons (0%)')
  })
  it('offers only complete runs of the same benchmark once one is chosen', () => {
    const runs = [
      { run_id: 'a', created_at: '', status: 'complete', dataset: { sha256: 'x' } },
      { run_id: 'b', created_at: '', status: 'complete', dataset: { sha256: 'y' } },
      { run_id: 'c', created_at: '', status: 'partial', dataset: { sha256: 'x' } },
      { run_id: 'd', created_at: '', status: 'complete', dataset: { sha256: 'x' } },
    ]
    expect(choosableRuns(runs, []).map(r => r.run_id)).toEqual(['a', 'b', 'd'])
    expect(choosableRuns(runs, ['a']).map(r => r.run_id)).toEqual(['a', 'd'])
  })
  it('states each proposed change with the noise behind it', () => {
    expect(changeLine({ path: 'metrics.ndcg.max_drop', from: 0.001, to: 0.015, observed: 0.0098 })).toBe('metrics.ndcg.max_drop: 0.0010 → 0.015 (observed 0.0098)')
    expect(changeLine({ path: 'max_regressed_queries', from: 0, to: 3, observed: 2 })).toBe('max_regressed_queries: 0 → 3 (observed 2)')
    expect(changeLine({ path: 'latency.min_increase_ms', from: null, to: 4.5, observed: 3 })).toBe('latency.min_increase_ms: not set → 4.5 (observed 3)')
  })
})
