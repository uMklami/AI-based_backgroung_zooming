/* AI Selective Zoom — service worker
   Cache-first for the app shell; network-only for processing/API routes. */
const CACHE = "szoom-v1";
const SHELL = [
  "/",
  "/static/CSS/Style.css",
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png"
];
const NETWORK_ONLY = [
  "/process", "/status", "/segment", "/segment_init", "/uploads", "/results"
];

self.addEventListener("install", e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;                       // CDNs: browser default
  if (NETWORK_ONLY.some(p => url.pathname.startsWith(p))) return;    // API: always network

  e.respondWith(
    caches.match(e.request).then(hit =>
      hit ||
      fetch(e.request).then(resp => {
        const copy = resp.clone();
        caches.open(CACHE).then(c => c.put(e.request, copy));
        return resp;
      }).catch(() => caches.match("/"))
    )
  );
});
