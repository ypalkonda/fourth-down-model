import { ACTION, ACTIONS, pct } from '../lib.js'

/*
  Three options, one measure. Every bar carries a visible value label, which
  is also what discharges the contrast relief rule for the aqua slot on the
  light surface — the number is readable even where the fill is not.

  The bars are scaled to the largest option rather than to 1.0. Win
  probabilities for three alternatives in the same situation usually sit
  within a few points of each other, and a 0-100% axis would flatten the
  differences the chart exists to show. The axis note states the scaling so
  it cannot be misread as absolute.
*/
export default function ActionBars({ wp, ep, recommendation, mode = 'wp' }) {
  const values = mode === 'wp' ? wp : ep
  const max = Math.max(...ACTIONS.map((a) => values[a]))
  const min = Math.min(...ACTIONS.map((a) => values[a]), 0)
  const span = Math.max(max - min, 1e-6)

  return (
    <div>
      {ACTIONS.map((a) => {
        const v = values[a]
        const width = `${Math.max(2, ((v - min) / span) * 100)}%`
        const isRec = a === recommendation
        return (
          <div className="optrow" key={a}>
            <div className="optname">
              <span className="swatch" style={{ background: ACTION[a].css }} />
              {ACTION[a].label}
            </div>
            <div className="bartrack">
              <div
                className="barfill"
                style={{
                  width,
                  background: ACTION[a].css,
                  opacity: isRec ? 1 : 0.42,
                }}
              />
            </div>
            <div className="optval" style={{ color: isRec ? ACTION[a].css : 'var(--text-secondary)' }}>
              {mode === 'wp' ? pct(v) : v.toFixed(2)}
            </div>
          </div>
        )
      })}
      <div className="card-note" style={{ marginTop: 8 }}>
        {mode === 'wp'
          ? 'Win probability for the possessing team. Bars are scaled to the range of the three options, not to 0–100%, so small but real edges stay visible.'
          : 'Expected points until the next score. Bars are scaled to the range of the three options.'}
      </div>
    </div>
  )
}
