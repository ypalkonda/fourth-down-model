import { useState } from 'react'
import { pct, useTooltip } from '../lib.js'
import Tooltip from './Tooltip.jsx'

/*
  Two series on one axis — both are the same measure (share of fourth downs
  gone for), so there is no temptation toward a second y-scale.

  The finding only reads correctly because every season is scored at one
  reference season: the optimal line is flat, and the movement is entirely in
  coach behaviour. Scoring each year under its own era features made the
  optimal line appear to rise, which was the model reading its own inputs.
*/

const ML = 46, MT = 18, MR = 74, MB = 40
const W = 720, H = 300

export default function TrendChart({ rows }) {
  const { tip, show, hide } = useTooltip()
  const [view, setView] = useState('chart')
  if (!rows?.length) return <div className="empty">Loading trend…</div>

  const xs = rows.map((r) => r.season)
  const x = (s) => ML + ((s - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - ML - MR)
  const hi = 0.4
  const y = (v) => H - MB - (v / hi) * (H - MT - MB)

  const series = [
    { key: 'optimal_go_rate', label: 'Model optimal', color: 'var(--punt)' },
    { key: 'actual_go_rate', label: 'Coaches actually', color: 'var(--go)' },
  ]
  const path = (k) => rows.map((r, i) => `${i ? 'L' : 'M'}${x(r.season)},${y(r[k])}`).join(' ')

  return (
    <div>
      <div className="card-head">
        <h2>Coaches are closing the gap, slowly</h2>
        <div className="viewtoggle">
          <button aria-pressed={view === 'chart'} onClick={() => setView('chart')}>Chart</button>
          <button aria-pressed={view === 'table'} onClick={() => setView('table')}>Table</button>
        </div>
      </div>
      <div className="legend" style={{ margin: '8px 0 12px' }}>
        {series.map((s) => (
          <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>
        ))}
      </div>

      {view === 'chart' ? (
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
          {[0, 0.1, 0.2, 0.3, 0.4].map((t) => (
            <g key={t}>
              <line x1={ML} y1={y(t)} x2={W - MR} y2={y(t)} stroke="var(--gridline)" strokeWidth="1" />
              <text x={ML - 9} y={y(t)} textAnchor="end" dominantBaseline="central"
                    fontSize="11" fill="var(--text-muted)">{Math.round(t * 100)}%</text>
            </g>
          ))}
          {rows.map((r) => (
            <text key={r.season} x={x(r.season)} y={H - MB + 18} textAnchor="middle"
                  fontSize="11" fill="var(--text-muted)">{String(r.season).slice(2)}</text>
          ))}

          {series.map((s) => (
            <g key={s.key}>
              <path d={path(s.key)} fill="none" stroke={s.color} strokeWidth="2"
                    strokeLinejoin="round" strokeLinecap="round" />
              {rows.map((r) => (
                <circle key={r.season} cx={x(r.season)} cy={y(r[s.key])} r="4"
                        fill={s.color} stroke="var(--surface)" strokeWidth="2" />
              ))}
              {/* direct label at the endpoint, so identity never rests on colour */}
              <text x={W - MR + 8} y={y(rows[rows.length - 1][s.key])} dominantBaseline="central"
                    fontSize="11.5" fontWeight="600" fill={s.color}>
                {pct(rows[rows.length - 1][s.key], 0)}
              </text>
            </g>
          ))}

          {/* one wide hit band per season drives the shared tooltip */}
          {rows.map((r, i) => {
            const bw = (W - ML - MR) / (rows.length - 1)
            return (
              <rect key={`h${r.season}`} x={x(r.season) - bw / 2} y={MT}
                    width={bw} height={H - MT - MB} fill="transparent"
                    onMouseMove={(e) => show(e, (
                      <>
                        <div className="tt-title">{r.season}</div>
                        <div className="tt-row"><span>Coaches went</span><b>{pct(r.actual_go_rate)}</b></div>
                        <div className="tt-row"><span>Model optimal</span><b>{pct(r.optimal_go_rate)}</b></div>
                        <div className="tt-row"><span>WP lost / decision</span><b>{(100 * r.mean_wp_loss).toFixed(2)} pp</b></div>
                        <div className="tt-row"><span>Decisions</span><b>{r.n.toLocaleString()}</b></div>
                      </>
                    ))}
                    onMouseLeave={hide} />
            )
          })}
        </svg>
      ) : (
        <div className="tablewrap">
          <table>
            <thead>
              <tr><th>Season</th><th className="num">Decisions</th><th className="num">Coaches went</th>
                  <th className="num">Model optimal</th><th className="num">WP lost / decision</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.season}>
                  <td>{r.season}</td>
                  <td className="num">{r.n.toLocaleString()}</td>
                  <td className="num">{pct(r.actual_go_rate)}</td>
                  <td className="num">{pct(r.optimal_go_rate)}</td>
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
