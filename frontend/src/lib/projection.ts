/** Library-map (UMAP) coordinates -> panel pixels. */
export type Bounds = { x0: number; x1: number; y0: number; y1: number }

/** UMAP coords -> pixels, fitted (aspect kept, y up) into width x height. Shared by canvas and SVG overlays. */
export function fitProjection(b: Bounds, width: number, height: number, pad: number) {
  const s = Math.min((width - pad * 2) / (b.x1 - b.x0 || 1), (height - pad * 2) / (b.y1 - b.y0 || 1))
  const ox = (width - (b.x1 - b.x0) * s) / 2, oy = (height - (b.y1 - b.y0) * s) / 2
  return {
    px: (x: number) => ox + (x - b.x0) * s,
    py: (y: number) => height - (oy + (y - b.y0) * s),
    // pixels -> map coords (for zoom / pan / hit-testing)
    ix: (X: number) => b.x0 + (X - ox) / s,
    iy: (Y: number) => b.y0 + (height - Y - oy) / s,
    scale: s,
  }
}

export function boundsOf(pts: Iterable<readonly [number, number]>): Bounds {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y
  }
  return { x0, x1, y0, y1 }
}
