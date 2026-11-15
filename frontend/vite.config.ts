import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8001',
      '/tools': 'http://localhost:8001',
      '/graph': 'http://localhost:8001',
      '/tracks': 'http://localhost:8001',
      '/events': 'http://localhost:8001',
      '/agent': 'http://localhost:8001',
      '/playback': 'http://localhost:8001',
    },
  },
})
