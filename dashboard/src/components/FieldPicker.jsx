import { useCallback, useRef } from 'react'
import { fieldPos } from '../lib.js'

/*
  Interactive field. Drag or click anywhere to set the line of scrimmage; the
  chains and the field-goal distance follow.

  Geometry: 10px per yard. The playing field runs x = 120 (the offense's own
  goal line) to x = 1120 (the opponent's goal line), with a 100px end zone at
  each end. `yardline_100` counts yards to the end zone being attacked, so it
  maps right-to-left: x = 1120 - 10 * yardline_100.

  The yellow line is the broadcast first-down convention — a positional
  annotation, not a data series, so it does not draw from the categorical
  palette.
*/

const PX = 10
const LEFT_GOAL = 120
const RIGHT_GOAL = 1120
const TOP = 46
const BOT = 236

const xOf = (yardline100) => RIGHT_GOAL - yardline100 * PX
const yardlineFromX = (x) =>
  Math.max(1, Math.min(99, Math.round((RIGHT_GOAL - x) / PX)))

export default function FieldPicker({ yardline100, ydstogo, onChange }) {
  const svgRef = useRef(null)
  const dragging = useRef(false)

  const setFromEvent = useCallback(
    (clientX) => {
      const svg = svgRef.current
      if (!svg) return
      const box = svg.getBoundingClientRect()
      const x = ((clientX - box.left) / box.width) * 1240
      onChange(yardlineFromX(x))
    },
    [onChange]
  )

  const onPointerDown = (e) => {
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    setFromEvent(e.clientX)
  }
  const onPointerMove = (e) => dragging.current && setFromEvent(e.clientX)
  const onPointerUp = (e) => {
    dragging.current = false
    e.currentTarget.releasePointerCapture?.(e.pointerId)
  }
  const onKeyDown = (e) => {
    const step = e.shiftKey ? 5 : 1
    if (e.key === 'ArrowLeft') { e.preventDefault(); onChange(Math.min(99, yardline100 + step)) }
    if (e.key === 'ArrowRight') { e.preventDefault(); onChange(Math.max(1, yardline100 - step)) }
  }

  const losX = xOf(yardline100)
  const firstDownYard = Math.max(0, yardline100 - ydstogo)
  const fdX = xOf(firstDownYard)
  const goalToGo = ydstogo >= yardline100
  const fgDistance = yardline100 + 17

  // Field numbers: 10..50..10 measured from each goal line.
  const numbers = []
  for (let y = 10; y <= 90; y += 10) {
    numbers.push({ x: LEFT_GOAL + y * PX, n: y <= 50 ? y : 100 - y })
  }

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 1240 300"
      role="slider"
      tabIndex={0}
      aria-label="Line of scrimmage"
      aria-valuemin={1}
      aria-valuemax={99}
      aria-valuenow={yardline100}
      aria-valuetext={`${fieldPos(yardline100)}, 4th and ${ydstogo}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onKeyDown={onKeyDown}
      style={{ width: '100%', display: 'block', cursor: 'ew-resize', touchAction: 'none', borderRadius: 8 }}
    >
      {/* turf */}
      <rect x={LEFT_GOAL} y={TOP} width={RIGHT_GOAL - LEFT_GOAL} height={BOT - TOP}
            fill="var(--wash)" />
      <rect x={20} y={TOP} width={100} height={BOT - TOP} fill="var(--gridline)" opacity=".7" />
      <rect x={RIGHT_GOAL} y={TOP} width={100} height={BOT - TOP} fill="var(--gridline)" opacity=".7" />

      <text x={70} y={(TOP + BOT) / 2} textAnchor="middle" dominantBaseline="central"
            fontSize="13" fontWeight="600" fill="var(--text-muted)"
            transform={`rotate(-90 70 ${(TOP + BOT) / 2})`} letterSpacing="2">
        OWN
      </text>
      <text x={1170} y={(TOP + BOT) / 2} textAnchor="middle" dominantBaseline="central"
            fontSize="13" fontWeight="600" fill="var(--text-muted)"
            transform={`rotate(90 1170 ${(TOP + BOT) / 2})`} letterSpacing="2">
        OPP
      </text>

      {/* five-yard lines */}
      {Array.from({ length: 21 }, (_, i) => LEFT_GOAL + i * 5 * PX).map((x, i) => (
        <line key={x} x1={x} y1={TOP} x2={x} y2={BOT}
              stroke="var(--baseline)" strokeWidth={i === 10 ? 1.5 : 1}
              opacity={i % 2 === 0 ? 0.85 : 0.4} />
      ))}

      {/* hash marks */}
      {Array.from({ length: 99 }, (_, i) => LEFT_GOAL + (i + 1) * PX).map((x) => (
        <g key={`h${x}`} stroke="var(--baseline)" strokeWidth="1" opacity=".45">
          <line x1={x} y1={TOP + 52} x2={x} y2={TOP + 60} />
          <line x1={x} y1={BOT - 60} x2={x} y2={BOT - 52} />
        </g>
      ))}

      {numbers.map(({ x, n }) => (
        <text key={`n${x}`} x={x} y={BOT - 12} textAnchor="middle"
              fontSize="15" fontWeight="600" fill="var(--text-muted)"
              fontVariant="tabular-nums">{n}</text>
      ))}

      {/* Yards-to-go band, drawn under the markers. The first-down marker is
          always downfield of the snap, and x decreases as yardline_100 grows,
          so the band runs losX -> fdX in that order. */}
      {!goalToGo && (
        <rect x={losX} y={TOP} width={Math.max(0, fdX - losX)} height={BOT - TOP}
              fill="var(--go)" opacity=".22" />
      )}

      {/* First-down line (broadcast yellow). The label is anchored to the
          right of the line rather than centred on it: on a short-yardage play
          the marker sits only a few pixels downfield of the ball, and a
          centred label lands underneath it. */}
      {!goalToGo && (
        <>
          <line x1={fdX} y1={TOP - 6} x2={fdX} y2={BOT + 6}
                stroke="var(--warning)" strokeWidth="2.5" />
          <text x={fdX + 6} y={TOP - 12} textAnchor="start" fontSize="11"
                fontWeight="650" fill="var(--warning)">TO GAIN</text>
        </>
      )}

      {/* line of scrimmage */}
      <line x1={losX} y1={TOP - 6} x2={losX} y2={BOT + 6}
            stroke="var(--text-primary)" strokeWidth="2.5" />
      <circle cx={losX} cy={TOP - 18} r="8" fill="var(--text-primary)" />
      <circle cx={losX} cy={TOP - 18} r="3.2" fill="var(--surface)" />

      {/* readouts */}
      <text x={losX} y={BOT + 32} textAnchor="middle" fontSize="15" fontWeight="650"
            fill="var(--text-primary)">
        {goalToGo ? `4th & goal` : `4th & ${ydstogo}`}
      </text>
      <text x={losX} y={BOT + 50} textAnchor="middle" fontSize="12.5"
            fill="var(--text-secondary)">
        {fieldPos(yardline100)} · {fgDistance}-yd attempt
      </text>
    </svg>
  )
}
