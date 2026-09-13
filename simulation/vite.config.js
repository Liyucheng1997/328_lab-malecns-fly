import { defineConfig } from 'vite';
import { createReadStream, statSync } from 'node:fs';
import { resolve, sep } from 'node:path';

// Vite's static middleware treats .gz as precompressed responses. The app
// verifies the compressed bytes and decompresses them itself, so serve raw gzip.
function rawGzip(root) {
  const base = resolve(root);
  return (req, res, next) => {
    const url = decodeURIComponent((req.url || '').split('?')[0]);
    if (!url.endsWith('.gz')) return next();
    const file = resolve(base, '.' + url);
    if (!file.startsWith(base + sep)) return next();
    try {
      const stat = statSync(file);
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Content-Length', stat.size);
      res.setHeader('Cache-Control', 'no-cache');
      createReadStream(file).pipe(res);
    } catch { next(); }
  };
}

export default defineConfig({
  base: './',
  worker: { format: 'es' },
  plugins: [{
    name: 'raw-connectome-gzip',
    configureServer(server) { server.middlewares.use(rawGzip('public')); },
    configurePreviewServer(server) { server.middlewares.use(rawGzip('dist')); },
  }],
});
