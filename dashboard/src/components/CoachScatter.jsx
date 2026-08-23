import { useState } from 'react'
import { pct, useTooltip } from '../lib.js'
import Tooltip from './Tooltip.jsx'

/*
  One series, so one colour and no legend — the title names it. The parity
  diagonal is the whole argument: a coach who went for it exactly as often as
  the model recommends would sit on the line. Every point falls below it.

  Marker area encodes sample size, which is a second variable rather than a
  restatement of position, and each dot carries an invisible 24px hit target
  so the hover does not demand pixel accuracy.
*/

const ML = 54, MT = 16, MR = 20, MB = 46
const W = 720, H = 520

export default function CoachScatter({ rows }) {
  const { tip, show, hide } = useTooltip()
  const [view, setView] = useState('chart')
  if (!rows?.length) return <div className="empty">Loading coaches…</div>

  const maxRate = Math.max(
    ...rows.map((r) => Math.max(r.actual_go_rate, r.optimal_go_rate))
  )
  const hi = Math.ceil((maxRate * 100) / 5) * 5 / 100 + 0.02
  const x = (v) => ML + (v / hi) * (W - ML - MR)
  const y = (v) => H - MB - (v / hi) * (H - MT - MB)
  const maxN = Math.max(...rows.map((r) => r.n))
  const rOf = (n) => 5 + 7 * Math.sqrt(n / maxN)

  const ticks = []
  for (let v = 0; v <= hi; v += 0.1) ticks.push(v)

  // Label the extremes only: most conservative, and closest to parity.
  const byIndex = [...rows].sort((a, b) => a.aggression_index - b.aggression_index)
  const labelled = [...byIndex.slice(0, 2), ...byIndex.slice(-3)]

  /* Nudge labels apart vertically. Two of the most conservative coaches sit
     almost on top of each other, and their names overlapped into an unreadable
     smear. Anything within 15px of the previous label gets pushed down. */
  const placed = labelled
    .map((r) => ({ r, x: x(r.optimal_go_rate), y: y(r.actual_go_rate) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < placed.length; i++) {
    const gap = placed[i].y - placed[i - 1].y
    if (gap < 15) placed[i].y = placed[i - 1].y + 15
  }

  return (
    <div>
      <div className="card-head">
        <h2>Every coach is more conservative than the model</h2>
        <div className="viewtoggle">
          <button aria-pressed={view === 'chart'} onClick={() => setView('chart')}>Chart</button>
          <button aria-pressed={view === 'table'} onClick={() => setView('table')}>Table</button>
        </div>
      </div>
      <div className="sub" style={{ marginBottom: 12 }}>
        {rows.length} coaches with 150+ meaningful fourth downs, 2016–2025. Marker size = number of decisions.
      </div>

      {view === 'chart' ? (
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={ML} y1={y(t)} x2={W - MR} y2={y(t)} stroke="var(--gridline)" strokeWidth="1" />
              <line x1={x(t)} y1={MT} x2={x(t)} y2={H - MB} stroke="var(--gridline)" strokeWidth="1" />
              <text x={ML - 9} y={y(t)} textAnchor="end" dominantBaseline="central"
                    fontSize="11" fill="var(--text-muted)">{Math.round(t * 100)}%</text>
              <text x={x(t)} y={H - MB + 18} textAnchor="middle"
                    fontSize="11" fill="var(--text-muted)">{Math.round(t * 100)}%</text>
            </g>
          ))}

          {/* Parity. The label rides the line at two-thirds rather than its
              end, where it was running off the right edge of the plot. */}
          <line x1={x(0)} y1={y(0)} x2={x(hi)} y2={y(hi)}
                stroke="var(--text-muted)" strokeWidth="1.5" opacity=".55" />
          <text x={x(hi * 0.66)} y={y(hi * 0.66) - 9} textAnchor="middle" fontSize="11"
                fill="var(--text-muted)">as aggressive as the model</text>

          {rows.map((r) => (
            <g key={r.coach}>
              <circle cx={x(r.optimal_go_rate)} cy={y(r.actual_go_rate)} r="14" fill="transparent"
                      onMouseMove={(e) => show(e, (
                        <>
                          <div className="tt-title">{r.coach}</div>
                          <div className="tt-row"><span>Went for it</span><b>{pct(r.actual_go_rate)}</b></div>
                          <div className="tt-row"><span>Model says</span><b>{pct(r.optimal_go_rate)}</b></div>
                          <div className="tt-row"><span>Decisions</span><b>{r.n.toLocaleString()}</b></div>
                          <div className="tt-row"><span>WP lost / decision</span><b>{(100 * r.mean_wp_loss).toFixed(2)} pp</b></div>
                        </>
                      ))}
                      onMouseLeave={hide} />
              <circle cx={x(r.optimal_go_rate)} cy={y(r.actual_go_rate)} r={rOf(r.n)}
                      fill="var(--go)" fillOpacity=".62"
                      stroke="var(--surface)" strokeWidth="2" pointerEvents="none" />
            </g>
          ))}

          {placed.map(({ r, x: px, y: py }) => {
            const ly = y(r.actual_go_rate)
            return (
              <g key={`l${r.coach}`} pointerEvents="none">
                {/* leader line where the label had to be nudged off its dot */}
                {Math.abs(py - ly) > 2 && (
                  <line x1={px + rOf(r.n)} y1={ly} x2={px + rOf(r.n) + 6} y2={py}
                        stroke="var(--text-muted)" strokeWidth="1" opacity=".5" />
                )}
                <text x={px + rOf(r.n) + 8} y={py} dominantBaseline="central"
                      fontSize="11.5" fill="var(--text-secondary)">
                  {r.coach}
                </text>
              </g>
            )
          })}

          <text x={ML + (W - ML - MR) / 2} y={H - 6} textAnchor="middle"
                fontSize="12" fill="var(--text-secondary)">
            share of fourth downs the model would go for
          </text>
          <text x={15} y={MT + (H - MT - MB) / 2} textAnchor="middle" fontSize="12"
                fill="var(--text-secondary)"
                transform={`rotate(-90 15 ${MT + (H - MT - MB) / 2})`}>
            share the coach actually went for
          </text>
        </svg>
      ) : (
        <div className="tablewrap" style={{ maxHeight: 460, overflowY: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Coach</th><th className="num">n</th><th className="num">Went</th>
                <th className="num">Model</th><th className="num">Index</th><th className="num">WP lost</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.coach}>
                  <td>{r.coach}</td>
                  <td className="num">{r.n.toLocaleString()}</td>
                  <td className="num">{pct(r.actual_go_rate)}</td>
                  <td className="num">{pct(r.optimal_go_rate)}</td>
                  <td className="num">{r.aggression_index.toFixed(2)}</td>
                  <td className="num">{(100 * r.mean_wp_loss).toFixed(2)} pp</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Tooltip tip={tip} />
    </div>
  )
}
