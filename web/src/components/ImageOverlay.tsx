import { useMemo, useState } from "react";
import { colorFor, type Shape } from "../geometry";

const GROUP_COLORS: Record<string, string> = {
  lines: "hsl(217 90% 55%)",
  layout: "hsl(28 95% 52%)",
};

type Props = { src: string; shapes: Shape[] };

/** The input image with the result's boxes and outlines drawn over it. */
export function ImageOverlay({ src, shapes }: Props) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const groups = useMemo(() => [...new Set(shapes.map((s) => s.group))], [shapes]);
  const [hidden, setHidden] = useState<Set<string>>(() => new Set(groups.includes("lines") ? ["layout"] : []));
  const [labels, setLabels] = useState(true);

  const visible = shapes.map((s, i) => ({ s, i })).filter(({ s }) => !hidden.has(s.group) && (s.page ?? 1) === 1);
  const stroke = size ? Math.max(1.5, Math.min(size.w, size.h) / 400) : 2;
  const font = size ? Math.max(11, Math.min(size.w, size.h) / 50) : 12;

  return (
    <div className="overlay">
      <div className="overlay-controls">
        {groups.length > 1 &&
          groups.map((g) => (
            <label key={g} className="check">
              <input
                type="checkbox"
                checked={!hidden.has(g)}
                onChange={() =>
                  setHidden((prev) => {
                    const next = new Set(prev);
                    if (next.has(g)) next.delete(g);
                    else next.add(g);
                    return next;
                  })
                }
              />
              {g}
            </label>
          ))}
        <label className="check">
          <input type="checkbox" checked={labels} onChange={(e) => setLabels(e.target.checked)} />
          labels
        </label>
        <span className="muted">{visible.length} shown</span>
      </div>
      <div className="overlay-stage">
        <img src={src} alt="" onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
        {size && (
          <svg viewBox={`0 0 ${size.w} ${size.h}`} preserveAspectRatio="none">
            {visible.map(({ s, i }) => {
              const color = GROUP_COLORS[s.group] ?? colorFor(s.label);
              const active = hover === i;
              const pts = (s.polygon ?? s.corners).map((p) => `${p.x},${p.y}`).join(" ");
              const box = s.corners.map((p) => `${p.x},${p.y}`).join(" ");
              return (
                <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                  {s.polygon && <polygon points={pts} fill={color} fillOpacity={active ? 0.35 : 0.18} stroke="none" />}
                  <polygon
                    points={box}
                    fill={s.polygon ? "none" : color}
                    fillOpacity={active ? 0.25 : 0.06}
                    stroke={color}
                    strokeWidth={active ? stroke * 2 : stroke}
                  />
                  {(labels || active) && s.label && (s.group === "boxes" || active) && (
                    <text
                      x={s.corners[0].x}
                      y={Math.max(font, s.corners[0].y - stroke * 2)}
                      fontSize={font}
                      fill="white"
                      stroke={color}
                      strokeWidth={font / 4}
                      paintOrder="stroke"
                    >
                      {s.label.length > 60 ? `${s.label.slice(0, 57)}…` : s.label}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      {hover !== null && shapes[hover] && <div className="overlay-hover">{shapes[hover].label || "(no label)"}</div>}
    </div>
  );
}
