import axios from 'axios'

const client = axios.create({
  baseURL: '/api/v1', // dev — через прокси Vite; prod — через nginx reverse-proxy
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

client.interceptors.response.use(
  (r) => r,
  (error) => {
    // FastAPI 422 кладёт детали в error.response.data.detail
    console.error('API Error:', error.response?.data ?? error.message)
    return Promise.reject(error)
  },
)

export default client
