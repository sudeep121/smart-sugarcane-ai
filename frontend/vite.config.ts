import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Recharts is by far the largest dependency; splitting it keeps the
        // initial bundle small for pages that do not render a chart.
        manualChunks: {
          react: ['react', 'react-dom', 'react-router-dom'],
          charts: ['recharts'],
        },
      },
    },
  },
  server: {
    port: 5173,
    open: true,
    // Listen on every interface, not just loopback, so a phone on the same
    // Wi-Fi can open http://<this-machine-ip>:5173. Vite binds localhost by
    // default, which no other device can resolve to this computer.
    host: true,
    proxy: {
      // Uploaded images are served by FastAPI at /uploads/... - proxying them in
      // dev means <img src="/uploads/..."> works without hard-coding the host.
      // 127.0.0.1, not localhost: Node also prefers IPv6 for `localhost`, while
      // uvicorn binds IPv4 only by default, which makes the proxy fail.
      '/uploads': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
