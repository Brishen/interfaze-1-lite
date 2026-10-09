/** Boxes and outlines found anywhere in a tool's result, in the input's pixels. */

export type Point = { x: number; y: number };

export type Shape = {
  label: string;
  /** Four corners, clockwise from the top left. */
  corners: Point[];
  polygon?: Point[];
  page?: number;
  group: string;
};

function isPoint(v: unknown): v is Point {
  return !!v && typeof v === "object" && typeof (v as Point).x === "number" && typeof (v as Point).y === "number";
}

function cornersOf(bounds: unknown): Point[] | null {
  if (!bounds || typeof bounds !== "object") return null;
  const b = bounds as Record<string, unknown>;
  const tl = b.top_left, tr = b.top_right, br = b.bottom_right, bl = b.bottom_left;
  if (isPoint(tl) && isPoint(br)) {
    return [tl, isPoint(tr) ? tr : { x: br.x, y: tl.y }, br, isPoint(bl) ? bl : { x: tl.x, y: br.y }];
  }
  return null;
}

function polygonOf(v: unknown): Point[] | undefined {
  if (!Array.isArray(v) || v.length < 3) return undefined;
  const pts = v.map((p) => (Array.isArray(p) ? { x: Number(p[0]), y: Number(p[1]) } : isPoint(p) ? p : null));
  return pts.every((p): p is Point => !!p && Number.isFinite(p.x) && Number.isFinite(p.y)) ? pts : undefined;
}

/**
 * Walk a result for objects carrying `bounds`. The walk stops at the first boxed object
 * on each branch, so an OCR line is drawn once rather than once more per word.
 */
export function collectShapes(result: unknown): Shape[] {
  const out: Shape[] = [];
  const walk = (node: unknown, group: string, page?: number) => {
    if (Array.isArray(node)) {
      node.forEach((n) => walk(n, group, page));
      return;
    }
    if (!node || typeof node !== "object") return;
    const obj = node as Record<string, unknown>;
    const here = typeof obj.page === "number" ? obj.page : page;
    const corners = cornersOf(obj.bounds);
    if (corners) {
      const label = [obj.label, obj.type, obj.text].find((v) => typeof v === "string" && v) as string | undefined;
      out.push({ label: label ?? "", corners, polygon: polygonOf(obj.polygon), page: here, group });
      return;
    }
    for (const [key, value] of Object.entries(obj)) {
      if (key === "mask" || key === "words") continue;
      walk(value, key === "lines" || key === "layout" ? key : group, here);
    }
  };
  walk(result, "boxes");
  return out;
}

/** A stable, readable colour for a label. */
export function colorFor(label: string): string {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360} 85% 50%)`;
}
