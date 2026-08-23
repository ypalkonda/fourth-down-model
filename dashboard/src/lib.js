import { useEffect, useRef, useState } from 'react'

/* Action identity is fixed here and used everywhere. Colour follows the
   entity, never its rank, so "go is blue" survives every filter and sort. */
export const ACTION = {
  go:         { label: 'Go for it',  short: 'GO',   css: 'var(--go)' },
  punt:       { label: 'Punt',       short: 'PUNT', css: 'var(--punt)' },
  field_goal: { label: 'Field goal', short: 'FG',   css: 'var(--fg)' },
}
export const ACTIONS = ['go', 'punt', 'field_goal']

export const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`
export const pp = (x, d = 1) => `${(100 * x).toFixed(d)} pp`
export const signed = (x, d = 2) => `${x >= 0 ? '+' : ''}${x.toFixed(d)}`

/** "own 34" / "opp 12" / "midfield" — matches the engine's own phrasing. */
export const fieldPos = (yardline100) =>
  yardline100 === 50 ? 'midfield'
    : yardline100 > 50 ? `own ${100 - yardline100}`
    : `opp ${yardline100}`

export const clockLabel = (secs) =>
  `${Math.floor(secs / 60)}:${String(Math.round(secs) % 60).padStart(2, '0')}`

/** Debounced value — keeps slider drags from firing a request per pixel. */
export function useDebounced(value, ms = 180) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

/**
 * Async data with stale-while-revalidating semantics: the previous payload
 * stays on screen (dimmed by the caller) instead of collapsing to a skeleton,
 * so dragging a slider never causes a layout jump.
 */
export function useAsync(fn, deps, initial = null) {
  const [data, setData] = useState(initial)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const seq = useRef(0)

  useEffect(() => {
    const mine = ++seq.current
    setLoading(true)
    fn()
      .then((d) => { if (mine === seq.current) { setData(d); setError(null) } })
      .catch((e) => { if (mine === seq.current) setError(e) })
      .finally(() => { if (mine === seq.current) setLoading(false) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, loading, error }
}

/** Shared hover-tooltip state for the SVG charts. */
export function useTooltip() {
  const [tip, setTip] = useState(null)
  const show = (e, content) =>
    setTip({ x: e.clientX, y: e.clientY, content })
  const hide = () => setTip(null)
  return { tip, show, hide }
}

export const linspace = (a, b, n) =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
