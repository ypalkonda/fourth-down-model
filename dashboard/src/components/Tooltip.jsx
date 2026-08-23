/** Fixed-position hover card. Offsets itself away from the viewport edges so
 *  it never gets clipped at the right or bottom of the window. */
export default function Tooltip({ tip }) {
  if (!tip) return null
  const flipX = tip.x > window.innerWidth - 220
  const flipY = tip.y > window.innerHeight - 170
  return (
    <div
      className="tooltip"
      style={{
        left: flipX ? undefined : tip.x + 14,
        right: flipX ? window.innerWidth - tip.x + 14 : undefined,
        top: flipY ? undefined : tip.y + 14,
        bottom: flipY ? window.innerHeight - tip.y + 14 : undefined,
      }}
    >
      {tip.content}
    </div>
  )
}
