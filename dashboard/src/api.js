const j = async (r) => {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

export const getMeta = () => fetch('/api/meta').then(j)

export const evaluate = (scenario) =>
  fetch('/api/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(scenario),
  }).then(j)

export const getGrid = (scenario) =>
  fetch('/api/grid?' + new URLSearchParams(scenario)).then(j)

export const getBlunders = (filters) =>
  fetch('/api/blunders?' + new URLSearchParams(filters)).then(j)

export const getCoaches = () => fetch('/api/coaches').then(j)
export const getSeasons = () => fetch('/api/seasons').then(j)
