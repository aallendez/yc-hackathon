import { useEffect, useState } from 'react'

const cache = new Map<string, Promise<string | null>>()

function fetchCover(id: string): Promise<string | null> {
  let p = cache.get(id)
  if (!p) {
    p = fetch(`/tracks/cover?id=${encodeURIComponent(id)}`)
      .then((r) => (r.ok ? r.json() : { url: null }))
      .then((d) => d.url ?? null)
      .catch(() => null)
    cache.set(id, p)
  }
  return p
}

/** Album art URL for a track id (Spotify oEmbed via the backend), or null while loading / if none. */
export function useCover(id: string | null | undefined): string | null {
  const [url, setUrl] = useState<{ id: string; url: string | null } | null>(null)
  useEffect(() => {
    if (!id) return
    let live = true
    fetchCover(id).then((u) => live && setUrl({ id, url: u }))
    return () => { live = false }
  }, [id])
  return url && url.id === id ? url.url : null
}
