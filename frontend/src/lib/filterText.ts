/** Parsing the Tuner's filtered-sweep inputs. A pass rate is the fraction that PASSES. */

export type FilterCondition = Record<string, string | number | boolean | (string | number | boolean)[] | Record<string, number>>

/** `0.001, 0.01, 0.1` -> numbers in (0, 1], or null if any entry is not one. */
export function parsePassRates(text: string): number[] | null {
  const parts = text
    .split(/[,\s]+/)
    .map((p) => p.trim())
    .filter(Boolean)
  if (parts.length === 0 || parts.length > 8) return null
  const rates = parts.map(Number)
  return rates.every((r) => Number.isFinite(r) && r > 0 && r <= 1) ? [...new Set(rates)] : null
}

/**
 * One metadata filter per line, written `key=value` with `;` between ANDed conditions.
 * A value that is valid JSON (a number, a list, a `{"gte": 2020}` range) is read as JSON;
 * anything else is a string. Returns null if a line cannot be read.
 */
export function parseFilterLines(text: string): FilterCondition[] | null {
  const out: FilterCondition[] = []
  for (const line of text.split('\n').map((l) => l.trim()).filter(Boolean)) {
    const cond: FilterCondition = {}
    for (const part of line.split(';').map((p) => p.trim()).filter(Boolean)) {
      const at = part.indexOf('=')
      if (at < 1) return null
      const raw = part.slice(at + 1)
      let value: FilterCondition[string]
      try {
        value = JSON.parse(raw)
      } catch {
        value = raw
      }
      cond[part.slice(0, at).trim()] = value
    }
    if (Object.keys(cond).length === 0) return null
    out.push(cond)
  }
  return out.length > 4 ? null : out
}
