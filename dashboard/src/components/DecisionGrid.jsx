import { useMemo, useState } from 'react'
import { ACTION, fieldPos, pct } from '../lib.js'
import Tooltip from './Tooltip.jsx'
import { useTooltip } from '../lib.js'

/*
  The whole strategy in one image: for every combination of distance and field
  position, which call wins.

  Cell colour is the recommended action — a nominal category, so it takes
  three fixed categorical hues rather than a value ramp. Confidence is carried
  by fill opacity, a genuinely separate variable, and toss-ups drop to neutral
  grey instead of being forced into a colour the model cannot justify.

  Orientation matches the field above: own goal line at the left, the
  opponent's end zone at the right.
*/

const ML = 46, MT = 10, MR = 12, MB = 40
const CELL_H = 21

export default function DecisionGrid({ grid, stale }) {
  const { tip, show, hide } = useTooltip()
  const [view, setView] = useState('chart')

  const { togos, yards, lookup } = useMemo(() => {
    if (!grid) return { togos: [], yards: [], lookup: new Map() }
    const m = new Map()
    for (const c of grid.cells) m.set(`${c.togo}:${c.yardline_100}`, c)
    return { togos: grid.togos, yards: grid.yards, lookup: m }
  }, [grid])

  if (!grid) return <div className="empty">Computing decision grid…</div>

  const nx = yards.length
  const cw = 20
  const W = ML + nx * cw + MR
  const H = MT + togos.length * CELL_H + MB

  // Left-to-right = own goal to opponent goal, matching the field picker.
  const xAt = (i) => ML + (nx - 1 - i) * cw

  const xTicks = [90, 70, 50, 30, 10].map((yl) => {
    const i = yards.indexOf(yl)
    return i < 0 ? null : { x: xAt(i) + cw / 2, label: fieldPos(yl) }
  }).filter(Boolean)

  return (
    <div>
      <div className="card-head">
        <h2>Decision grid</h2>
        <span className="sub">every distance × field position</span>
        <div className="viewtoggle">
          <button aria-pressed={view === 'chart'} onClick={() => setView('chart')}>Chart</button>
          <button aria-pressed={view === 'table'} onClick={() => setView('table')}>Table</button>
        </div>
      </div>

      <div className="legend" style={{ margin: '10px 0 14px' }}>
        {['go', 'punt', 'field_goal'].map((a) => (
          <span key={a}><i style={{ background: ACTION[a].css }} />{ACTION[a].label}</span>
        ))}
        <span><i style={{ background: 'var(--neutral)', opacity: .4 }} />Toss-up (within model error)</span>
      </div>

      {view === 'chart' ? (
        <div className={stale ? 'stale' : ''} style={{ overflowX: 'auto' }}>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: 680, display: 'block' }}>
            {togos.map((t, ti) => (
              <text key={`y${t}`} x={ML - 9} y={MT + ti * CELL_H + CELL_H / 2}
                    textAnchor="end" dominantBaseline="central"
                    fontSize="11" fill="var(--text-muted)"
                    style={{ fontVariantNumeric: 'tabular-nums' }}>{t}</text>
            ))}
            <text x={13} y={MT + (togos.length * CELL_H) / 2} fontSize="11"
                  fill="var(--text-secondary)" textAnchor="middle"
                  transform={`rotate(-90 13 ${MT + (togos.length * CELL_H) / 2})`}>
              yards to go
            </text>

            {togos.map((t, ti) =>
              yards.map((y, yi) => {
                const c = lookup.get(`${t}:${y}`)
                if (!c) return null
                const toss = c.confidence === 'toss-up'
                return (
                  <rect
                    key={`${t}-${y}`}
                    x={xAt(yi) + 1} y={MT + ti * CELL_H + 1}
                    width={cw - 2} height={CELL_H - 2} rx="2"
                    fill={toss ? 'var(--neutral)' : ACTION[c.rec].css}
                    opacity={toss ? 0.28 : c.confidence === 'lean' ? 0.55 : 0.92}
                    onMouseMove={(e) =>
                      show(e, (
                        <>
                          <div className="tt-title">4th &amp; {t} · {fieldPos(y)}</div>
                          <div className="tt-row"><span>Go</span><b>{pct(c.wp.go)}</b></div>
                          <div className="tt-row"><span>Punt</span><b>{pct(c.wp.punt)}</b></div>
                          <div className="tt-row"><span>Field goal</span><b>{pct(c.wp.field_goal)}</b></div>
                          <div className="tt-row" style={{ marginTop: 5, paddingTop: 5, borderTop: '1px solid var(--border)' }}>
                            <span>{toss ? 'Toss-up' : ACTION[c.rec].label}</span>
                            <b>+{(100 * c.margin).toFixed(1)} pp</b>
                          </div>
                        </>
                      ))
                    }
                    onMouseLeave={hide}
                  />
                )
              })
            )}

            {xTicks.map((t) => (
              <text key={t.label} x={t.x} y={H - MB + 22} textAnchor="middle"
                    fontSize="11" fill="var(--text-muted)">{t.label}</text>
            ))}
            <line x1={ML} y1={MT + togos.length * CELL_H + 4}
                  x2={ML + nx * cw} y2={MT + togos.length * CELL_H + 4}
                  stroke="var(--baseline)" strokeWidth="1" />
          </svg>
        </div>
      ) : (
        <GridTable togos={togos} yards={yards} lookup={lookup} />
      )}

      <div className="card-note">
        Reading left to right is driving downfield. Opacity marks confidence:
        solid where the edge is clear, faded where it is slim, neutral grey
        where the top two options sit within the model's own calibration error
        and it has no honest preference.
      </div>
      <Tooltip tip={tip} />
    </div>
  )
}

/** Table twin: contiguous field-position runs per recommendation. More
 *  readable than 588 individual cells, and it is the WCAG-clean equivalent. */
function GridTable({ togos, yards, lookup }) {
  const sorted = [...yards].sort((a, b) => b - a) // own goal -> opponent goal
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr><th>To go</th><th>Recommended, by field position</th></tr>
        </thead>
        <tbody>
          {togos.map((t) => {
            const runs = []
            for (const y of sorted) {
              const c = lookup.get(`${t}:${y}`)
              if (!c) continue
              const key = c.confidence === 'toss-up' ? 'tossup' : c.rec
              const last = runs[runs.length - 1]
              if (last && last.key === key) last.end = y
              else runs.push({ key, start: y, end: y })
            }
            return (
              <tr key={t}>
                <td className="num" style={{ width: 60 }}>{t}</td>
                <td>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {runs.map((r, i) => (
                      <span key={i} className="pill" style={{
                        color: r.key === 'tossup' ? 'var(--text-muted)' : ACTION[r.key].css,
                      }}>
                        {r.key === 'tossup' ? 'toss-up' : ACTION[r.key].short}
                        {' '}{fieldPos(r.start)}→{fieldPos(r.end)}
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
