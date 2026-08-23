import { ACTION, ACTIONS, pct } from '../lib.js'

/*
  Why the model says what it says: every outcome it considered, its
  probability, and what the resulting game state is worth. This is the part
  that makes the recommendation auditable instead of oracular.
*/
export default function BranchTree({ branches, recommendation }) {
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {ACTIONS.map((a) => {
        const rows = (branches?.[a] ?? []).slice(0, 6)
        if (!rows.length) return null
        const isRec = a === recommendation
        return (
          <div key={a}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 7, marginBottom: 7,
              fontSize: 12, fontWeight: 620,
              color: isRec ? ACTION[a].css : 'var(--text-secondary)',
            }}>
              <span className="swatch" style={{ background: ACTION[a].css }} />
              {ACTION[a].label}
              {isRec && <span style={{ fontSize: 10, letterSpacing: '.06em', color: 'var(--text-muted)' }}>
                RECOMMENDED
              </span>}
            </div>
            <table style={{ fontSize: 12 }}>
              <tbody>
                {rows.map((b, i) => (
                  <tr key={i}>
                    <td style={{ padding: '4px 8px 4px 0', border: 'none', color: 'var(--text-secondary)' }}>
                      {b.label}
                    </td>
                    <td className="num" style={{ padding: '4px 8px', border: 'none', width: 58, color: 'var(--text-muted)' }}>
                      {pct(b.prob, 1)}
                    </td>
                    <td style={{ padding: '4px 0', border: 'none', width: 110 }}>
                      <div style={{ height: 5, background: 'var(--wash)', borderRadius: 3 }}>
                        <div style={{
                          width: `${Math.max(1.5, b.prob * 100)}%`, height: '100%',
                          background: ACTION[a].css, opacity: isRec ? 0.9 : 0.4, borderRadius: 3,
                        }} />
                      </div>
                    </td>
                    <td className="num" style={{ padding: '4px 0 4px 8px', border: 'none', width: 54 }}>
                      {pct(b.wp, 1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      })}
      <div className="card-note">
        Each row is one outcome the engine enumerated: its probability, and the
        win probability of the game state it leads to. Where possession
        changes, that value is mirrored — your win probability is one minus
        theirs. Top six outcomes per option shown.
      </div>
    </div>
  )
}
