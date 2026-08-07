import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-прокси: фронт ходит на /api/*, Vite проксирует на FastAPI :8000.
// Это снимает проблему CORS на разработке (origin localhost:3000 в backend
// больше не критичен — запросы идут с того же origin, что и фронт).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
