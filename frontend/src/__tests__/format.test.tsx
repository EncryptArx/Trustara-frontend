import { describe, it, expect } from 'vitest'
import { detectDeepfake } from '../components/mockApi'

describe('mockApi and formatting', ()=>{
  it('returns the exact timestamp string and geo-tag', async ()=>{
    const res = await detectDeepfake()
    expect(res.timestamp).toBe('2025-11-15T18:03:22Z')
    expect(res.geo_tag).toBe('Bengaluru, India')
  })
})
