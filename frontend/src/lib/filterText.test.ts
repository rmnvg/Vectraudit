import { describe, expect, it } from 'vitest'

import { parseFilterLines, parsePassRates } from './filterText'

describe('parsePassRates', () => {
  it('reads fractions that pass', () => {
    expect(parsePassRates('0.5, 0.1 0.01')).toEqual([0.5, 0.1, 0.01])
    expect(parsePassRates('1')).toEqual([1])
  })
  it('refuses anything that is not a fraction in (0, 1]', () => {
    for (const bad of ['', '0', '-0.1', '1.5', '10%', 'a', '0.1,,x']) expect(parsePassRates(bad)).toBeNull()
    expect(parsePassRates('0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')).toBeNull()
  })
  it('drops duplicates', () => expect(parsePassRates('0.1, 0.1')).toEqual([0.1]))
})

describe('parseFilterLines', () => {
  it('reads one filter per line, with ; joining conditions', () => {
    expect(parseFilterLines('lang=de\nyear={"gte":2020};tag=["a","b"]')).toEqual([
      { lang: 'de' },
      { year: { gte: 2020 }, tag: ['a', 'b'] },
    ])
  })
  it('reads numbers as numbers and other text as strings', () => {
    expect(parseFilterLines('year=2024\nname=O\'Brien')).toEqual([{ year: 2024 }, { name: "O'Brien" }])
  })
  it('refuses a line it cannot read, and more than four filters', () => {
    expect(parseFilterLines('nonsense')).toBeNull()
    expect(parseFilterLines('=x')).toBeNull()
    expect(parseFilterLines('a=1\nb=1\nc=1\nd=1\ne=1')).toBeNull()
  })
})
