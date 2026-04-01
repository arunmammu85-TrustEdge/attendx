self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => clients.claim());
// No caching — always fresh from network
self.addEventListener('fetch', e => {
  e.respondWith(fetch(e.request));
});
