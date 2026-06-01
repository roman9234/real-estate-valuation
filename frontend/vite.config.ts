import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-прокси: фронт ходит на /api/*, Vite проксирует на FastAPI :8000.
// Это снимает проблему CORS на разработке (origin localhost:3000 в backend
// больше не критичен — запросы идут с того же origin, что и фронт).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // совпадает с allow_origins backend
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
