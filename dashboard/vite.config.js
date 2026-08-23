import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The Flask API runs separately in development; proxying keeps the
    // frontend on one origin so there is no CORS dance in the browser.
    proxy: { '/api': 'http://127.0.0.1:5001' },
  },
  build: { outDir: 'dist', sourcemap: false },
})
