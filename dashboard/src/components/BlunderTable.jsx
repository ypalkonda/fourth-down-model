import { useState } from 'react'
import { ACTION, pct } from '../lib.js'

/* Ranked fourth-down mistakes. Only meaningful, non-toss-up decisions are
   eligible: a model that ranks calls it cannot actually separate is just
   publishing its own noise with a confident label. */
export default function BlunderTable({ rows, total, stale }) {
  const [open, setOpen] = useState(null)
  if (!rows) return <div className="empty">Loading…</div>
  if (!rows.length) return <div className="empty">No decisions match these filters.</div>

  const worst = Math.max(...rows.map((r) => r.wp_loss))

  return (
    <div className={stale ? 'stale' : ''}>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th className="num" style={{ width: 128 }}>WP cost</th>
              <th style={{ width: 62 }}>Season</th>
              <th style={{ width: 108 }}>Matchup</th>
              <th style={{ width: 118 }}>Situation</th>
              <th style={{ width: 72 }}>Score</th>
              <th>Did</th>
              <th>Should</th>
              <th>Coach</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const id = `${r.season}-${r.posteam}-${i}`
              const isOpen = open === id
              return (
                <>
                  <tr key={id} onClick={() => setOpen(isOpen ? null : id)}
                      style={{ cursor: 'pointer' }}>
                    <td className="num">
                      <div style={{ display: 'flex', alignItems: 'center', gap: 9, justifyContent: 'flex-end' }}>
                        <div style={{ flex: 1, height: 8, background: 'var(--wash)', borderRadius: 4 }}>
                          <div style={{
                            width: `${(r.wp_loss / worst) * 100}%`, height: '100%',
                            background: 'var(--critical)', borderRadius: 4,
                          }} />
                        </div>
                        <b style={{ fontVariantNumeric: 'tabular-nums', fontSize: 14, minWidth: 34 }}>
                          {(100 * r.wp_loss).toFixed(1)}
                        </b>
                      </div>
                    </td>
                    <td style={{ color: 'var(--text-secondary)' }}>{r.season} w{r.week}</td>
                    <td><b>{r.posteam}</b> <span style={{ color: 'var(--text-muted)' }}>v {r.defteam}</span></td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>
                      4th &amp; {r.ydstogo} · {r.field_position}
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Q{r.qtr} {r.clock}</div>
                    </td>
                    <td className="num" style={{ color: r.score_differential < 0 ? 'var(--critical)' : 'var(--text-secondary)' }}>
                      {r.score_differential > 0 ? '+' : ''}{r.score_differential}
                    </td>
                    <td><span className="pill" style={{ color: ACTION[r.actual].css }}>{ACTION[r.actual].short}</span></td>
                    <td><span className="pill" style={{ color: ACTION[r.recommended].css }}>{ACTION[r.recommended].short}</span></td>
                    <td style={{ color: 'var(--text-secondary)' }}>{r.coach ?? '—'}</td>
                  </tr>
                  {isOpen && (
                    <tr key={`${id}-d`}>
                      <td colSpan={8} style={{ background: 'var(--wash)', padding: '14px 16px' }}>
                        <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap', marginBottom: 10 }}>
                          {['go', 'punt', 'field_goal'].map((a) => (
                            <div key={a}>
                              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em' }}>
                                {ACTION[a].label}
                              </div>
                              <div style={{
                                fontSize: 19, fontWeight: 640,
                                color: a === r.recommended ? ACTION[a].css : 'var(--text-secondary)',
                              }}>
                                {pct(r[`wp_${a}`])}
                              </div>
                            </div>
                          ))}
                          <div>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em' }}>
                              Convert / FG make
                            </div>
                            <div style={{ fontSize: 19, fontWeight: 640, color: 'var(--text-secondary)' }}>
                              {pct(r.convert_prob, 0)} / {pct(r.fg_make_prob, 0)}
                            </div>
                          </div>
                        </div>
                        {r.desc && (
                          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                            {r.desc}
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="card-note">
        Showing {rows.length.toLocaleString()} of {total?.toLocaleString?.() ?? '—'} matching decisions,
        worst first. Click any row for the full breakdown. Cost is the win
        probability the model says the call gave away.
      </div>
    </div>
  )
}
