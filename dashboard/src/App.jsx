import { useCallback, useEffect, useMemo, useState } from 'react'
import * as api from './api.js'
import { ACTION, clockLabel, pct, useAsync, useDebounced } from './lib.js'
import FieldPicker from './components/FieldPicker.jsx'
import ActionBars from './components/ActionBars.jsx'
import BranchTree from './components/BranchTree.jsx'
import DecisionGrid from './components/DecisionGrid.jsx'
import CoachScatter from './components/CoachScatter.jsx'
import TrendChart from './components/TrendChart.jsx'
import BlunderTable from './components/BlunderTable.jsx'

const TABS = [
  ['lab', 'Decision Lab'],
  ['blunders', 'Blunders'],
  ['coaches', 'Coaches'],
  ['method', 'Method'],
]

const TAB_KEYS = TABS.map(([k]) => k)

export default function App() {
  // Views live in the URL hash, so a specific view is linkable and the back
  // button behaves. `?theme=` overrides the stored preference, which also
  // makes the render both themes without touching localStorage.
  const [tab, setTab] = useState(() => {
    const h = window.location.hash.replace('#', '')
    return TAB_KEYS.includes(h) ? h : 'lab'
  })
  const [theme, setTheme] = useState(() => {
    const q = new URLSearchParams(window.location.search).get('theme')
    if (q === 'dark' || q === 'light') return q
    return localStorage.getItem('fd-theme') || 'auto'
  })

  useEffect(() => {
    const onHash = () => {
      const h = window.location.hash.replace('#', '')
      if (TAB_KEYS.includes(h)) setTab(h)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const go = (k) => { window.location.hash = k; setTab(k) }

  useEffect(() => {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('fd-theme', theme)
  }, [theme])

  const { data: meta } = useAsync(api.getMeta, [])

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead-inner">
          <div className="wordmark">
            <b>FOURTH DOWN</b>
            <span>NFL decision model · 2016–2025</span>
          </div>
          <nav className="tabs">
            {TABS.map(([k, label]) => (
              <button key={k} className="tab" onClick={() => go(k)}
                      aria-current={tab === k ? 'page' : undefined}>
                {label}
              </button>
            ))}
          </nav>
          <button className="icon-btn" title="Toggle theme"
                  onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <circle cx="12" cy="12" r="4.5" />
              <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
            </svg>
          </button>
        </div>
      </header>

      {tab === 'lab' && <Lab />}
      {tab === 'blunders' && <Blunders meta={meta} />}
      {tab === 'coaches' && <Coaches />}
      {tab === 'method' && <Method meta={meta} />}
    </div>
  )
}

/* ========================================================================= */

/* The landing scenario is 4th & 1 at midfield, tied, midway through the
   fourth. It is the canonical case the analytics literature is unanimous on
   and the model calls it clearly, so the first thing anyone sees is a
   recommendation they can check against something. */
const DEFAULT = {
  yardline_100: 50,
  ydstogo: 1,
  score_differential: 0,
  qtr: 4,
  clock_seconds: 480,
  posteam_timeouts: 3,
  defteam_timeouts: 3,
}

function Lab() {
  const [s, setS] = useState(DEFAULT)
  const [mode, setMode] = useState('wp')
  const set = useCallback((k, v) => setS((p) => ({ ...p, [k]: v })), [])

  // Yards to go can never exceed the distance to the end zone.
  const ydstogo = Math.min(s.ydstogo, s.yardline_100)
  const scenario = useMemo(() => ({ ...s, ydstogo }), [s, ydstogo])
  const debounced = useDebounced(scenario, 140)

  const { data: res, loading } = useAsync(
    () => api.evaluate(debounced), [JSON.stringify(debounced)]
  )
  const { data: grid, loading: gridLoading } = useAsync(
    () => api.getGrid({
      score_differential: debounced.score_differential,
      clock_seconds: debounced.clock_seconds,
      qtr: debounced.qtr,
      posteam_timeouts: debounced.posteam_timeouts,
      defteam_timeouts: debounced.defteam_timeouts,
    }),
    [debounced.score_differential, debounced.clock_seconds, debounced.qtr,
     debounced.posteam_timeouts, debounced.defteam_timeouts]
  )

  const accent = res ? ACTION[res.recommendation].css : 'var(--neutral)'

  return (
    <div className="stack">
      <div className="card" style={{ padding: '18px 20px 8px' }}>
        <FieldPicker
          yardline100={s.yardline_100}
          ydstogo={ydstogo}
          onChange={(v) => set('yardline_100', v)}
        />
      </div>

      <div className="grid-2">
        <div className="stack">
          <div className="card">
            <div className="card-head"><h2>Situation</h2>
              <span className="sub">drag the field above, or tune here</span></div>
            <div className="stack" style={{ gap: 14, marginTop: 14 }}>
              <Slider label="Yards to go" value={ydstogo} min={1}
                      max={Math.min(25, s.yardline_100)}
                      onChange={(v) => set('ydstogo', v)} />
              <Slider label="Score differential" value={s.score_differential}
                      min={-21} max={21}
                      fmt={(v) => (v > 0 ? `up ${v}` : v < 0 ? `down ${-v}` : 'tied')}
                      onChange={(v) => set('score_differential', v)} />
              <div className="ctrl">
                <div className="ctrl-label"><span>Quarter</span></div>
                <div className="seg">
                  {[1, 2, 3, 4].map((q) => (
                    <button key={q} aria-pressed={s.qtr === q} onClick={() => set('qtr', q)}>Q{q}</button>
                  ))}
                </div>
              </div>
              <Slider label="Clock (in quarter)" value={s.clock_seconds} min={0} max={900} step={15}
                      fmt={clockLabel} onChange={(v) => set('clock_seconds', v)} />
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <Timeouts label="Your timeouts" value={s.posteam_timeouts}
                          onChange={(v) => set('posteam_timeouts', v)} />
                <Timeouts label="Their timeouts" value={s.defteam_timeouts}
                          onChange={(v) => set('defteam_timeouts', v)} />
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-head"><h2>Why</h2></div>
            <div style={{ marginTop: 12 }}>
              {res && <BranchTree branches={res.branches} recommendation={res.recommendation} />}
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="verdict" style={{ '--accent': accent }}>
            <div className="verdict-label">The model says</div>
            <div className="verdict-call">
              {res ? ACTION[res.recommendation].label.toUpperCase() : '…'}
            </div>
            {res && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <span className={`badge ${res.confidence === 'toss-up' ? 'tossup' : ''}`}>
                  {res.confidence === 'toss-up' ? 'Too close to call'
                    : res.confidence === 'lean' ? 'Slight edge' : 'Clear'}
                </span>
                <span className="verdict-why">
                  {res.confidence === 'toss-up'
                    ? `Top two options differ by ${(100 * res.wp_margin).toFixed(1)} pp — inside the model's own error.`
                    : `Worth ${(100 * res.wp_margin).toFixed(1)} pp of win probability over the next best option.`}
                </span>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-head">
              <h2>{mode === 'wp' ? 'Win probability' : 'Expected points'}</h2>
              <div className="viewtoggle">
                <button aria-pressed={mode === 'wp'} onClick={() => setMode('wp')}>Win prob</button>
                <button aria-pressed={mode === 'ep'} onClick={() => setMode('ep')}>Exp. points</button>
              </div>
            </div>
            <div className={loading && !res ? '' : ''} style={{ marginTop: 10 }}>
              {res && <ActionBars wp={res.wp} ep={res.ep} recommendation={res.recommendation} mode={mode} />}
            </div>
          </div>

          {res && (
            <div className="tiles">
              <Tile k="Convert chance" v={pct(res.convert_prob, 0)}
                    d={`on 4th & ${ydstogo}`} />
              <Tile k="Field goal" v={pct(res.fg_make_prob, 0)}
                    d={`${res.fg_distance.toFixed(0)}-yard attempt`} />
              <Tile k="Field position" v={res.field_position} d="line of scrimmage" />
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <DecisionGrid grid={grid} stale={gridLoading && !!grid} />
      </div>
    </div>
  )
}

function Slider({ label, value, min, max, step = 1, fmt, onChange }) {
  return (
    <div className="ctrl">
      <div className="ctrl-label"><span>{label}</span><b>{fmt ? fmt(value) : value}</b></div>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  )
}

function Timeouts({ label, value, onChange }) {
  return (
    <div className="ctrl">
      <div className="ctrl-label"><span>{label}</span></div>
      <div className="seg">
        {[0, 1, 2, 3].map((t) => (
          <button key={t} aria-pressed={value === t} onClick={() => onChange(t)}>{t}</button>
        ))}
      </div>
    </div>
  )
}

function Tile({ k, v, d }) {
  return <div className="tile"><div className="k">{k}</div><div className="v">{v}</div><div className="d">{d}</div></div>
}

/* ========================================================================= */

function Blunders({ meta }) {
  const [f, setF] = useState({ season: '', team: '', coach: '', recommended: '', limit: 60 })
  const debounced = useDebounced(f, 120)
  const { data, loading } = useAsync(
    () => api.getBlunders(Object.fromEntries(Object.entries(debounced).filter(([, v]) => v !== ''))),
    [JSON.stringify(debounced)]
  )
  const set = (k, v) => setF((p) => ({ ...p, [k]: v }))

  return (
    <div>
      <div className="filterbar">
        <div className="f">
          <label htmlFor="fs">Season</label>
          <select id="fs" value={f.season} onChange={(e) => set('season', e.target.value)}>
            <option value="">All seasons</option>
            {meta?.seasons?.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="f">
          <label htmlFor="ft">Team</label>
          <select id="ft" value={f.team} onChange={(e) => set('team', e.target.value)}>
            <option value="">All teams</option>
            {meta?.teams?.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="f" style={{ minWidth: 190 }}>
          <label htmlFor="fc">Coach</label>
          <select id="fc" value={f.coach} onChange={(e) => set('coach', e.target.value)}>
            <option value="">All coaches</option>
            {meta?.coaches?.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="f">
          <label htmlFor="fr">Model recommended</label>
          <select id="fr" value={f.recommended} onChange={(e) => set('recommended', e.target.value)}>
            <option value="">Any</option>
            <option value="go">Go for it</option>
            <option value="punt">Punt</option>
            <option value="field_goal">Field goal</option>
          </select>
        </div>
        <div className="f">
          <label htmlFor="fl">Show</label>
          <select id="fl" value={f.limit} onChange={(e) => set('limit', Number(e.target.value))}>
            {[25, 60, 150, 400].map((n) => <option key={n} value={n}>Top {n}</option>)}
          </select>
        </div>
      </div>
      <div className="card">
        <BlunderTable rows={data?.rows} total={data?.total} stale={loading && !!data} />
      </div>
    </div>
  )
}

/* ========================================================================= */

function Coaches() {
  const { data: coaches } = useAsync(api.getCoaches, [])
  const { data: seasons } = useAsync(api.getSeasons, [])
  return (
    <div className="stack">
      <div className="card"><CoachScatter rows={coaches?.rows} /></div>
      <div className="card"><TrendChart rows={seasons?.rows} /></div>
    </div>
  )
}

/* ========================================================================= */

function Method({ meta }) {
  const t = meta?.totals
  return (
    <div className="stack">
      {t && (
        <div className="tiles">
          <Tile k="Fourth downs scored" v={t.scored.toLocaleString()} d="2016–2025, incl. playoffs" />
          <Tile k="Ranked decisions" v={t.real_decisions.toLocaleString()} d="meaningful, non-toss-up" />
          <Tile k="Model agrees" v={pct(t.agreement, 0)} d="with the coach's actual call" />
          <Tile k="Avg cost" v={`${(100 * t.mean_wp_loss).toFixed(2)} pp`} d="win probability per decision" />
        </div>
      )}

      {meta?.by_recommendation && (
        <div className="card">
          <div className="card-head"><h2>Coaches are wrong in one direction only</h2></div>
          <div className="tablewrap" style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr><th>Model says</th><th className="num">Decisions</th>
                    <th className="num">Coach agreed</th><th className="num">WP lost / decision</th></tr>
              </thead>
              <tbody>
                {meta.by_recommendation.map((r) => (
                  <tr key={r.action}>
                    <td><span className="pill" style={{ color: ACTION[r.action].css }}>
                      {ACTION[r.action].label}</span></td>
                    <td className="num">{r.n.toLocaleString()}</td>
                    <td className="num"><b>{pct(r.agreement, 1)}</b></td>
                    <td className="num">{(100 * r.mean_wp_loss).toFixed(2)} pp</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card-note">
            When the model says punt, coaches punt almost always. When it says go, they
            mostly punt anyway — and that error costs roughly twenty times more.
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head"><h2>How it works</h2></div>
        <div className="prose" style={{ marginTop: 12 }}>
          <p>
            Play-by-play data records what coaches <em>did</em>, never what they
            <em> should</em> have done. There is no ground-truth label for a correct
            decision, so this is not a classification problem. Training a classifier
            on coach behaviour would just reproduce coach behaviour, which is the
            thing being measured.
          </p>
          <p>Instead, four models and some arithmetic:</p>
          <ul>
            <li><strong>State value</strong> — a multi-task network over ~384k plays predicting
              the next score (seven classes → expected points) and the eventual winner
              (→ win probability) from a shared representation.</li>
            <li><strong>Go</strong> — the full distribution of yards gained, not a convert/fail
              coin flip, because where a failed attempt <em>ends</em> changes what you hand over.</li>
            <li><strong>Punt</strong> — the distribution of where the receiving team actually
              starts, read from the next real snap rather than reconstructed from kick
              distance and return yards.</li>
            <li><strong>Field goal</strong> — a logistic model in distance with a per-kicker
              random effect, monotone by construction.</li>
          </ul>
          <p>
            Each action's value is the probability-weighted value of the states it
            leads to. Where possession changes, the value mirrors:
            <code>WP(us) = 1 − WP(them)</code>. The recommendation is the argmax.
          </p>

          <h3>Why some cells say "too close to call"</h3>
          <p>
            The win-probability model's calibration error is about {t ? '2.7' : '—'} points.
            Any recommendation resting on a thinner margin than that is noise wearing a
            confident face, so margins under {meta ? (100 * meta.tossup_margin).toFixed(1) : '0.5'} pp
            are reported as toss-ups and excluded from the leaderboard entirely —
            {t ? ` ${t.tossups.toLocaleString()} of ${t.scored.toLocaleString()} situations.` : ''}
          </p>

          <h3>What it does not know</h3>
          <ul>
            <li>Every decision is scored team-neutral and with a league-average kicker.
              A model that knew your offense was bad would excuse punting, which is a
              claim about the roster rather than about the call.</li>
            <li>Injuries, weather beyond wind and temperature, momentum, and whether
              the punter just shanked one.</li>
            <li>Field-goal accuracy on 55+ yard attempts is under-predicted by roughly
              10 points, because league-wide kicking improved faster than the training
              window captured. The error is conservative — it discourages marginal long
              attempts rather than encouraging them.</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
