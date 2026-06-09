import { Link, Route, Routes, useLocation } from 'react-router-dom'
import EstimatePage from './pages/EstimatePage'
import ExplainPage from './pages/ExplainPage'

export default function App() {
  const { pathname } = useLocation()
  return (
    <>
      <header>
        <div>
          <h1>Оценка стоимости квартир · Москва</h1>
          <p>Вторичный рынок · машинное обучение с интерпретацией</p>
        </div>
        <nav>
          <Link to="/" className={pathname === '/' ? 'active' : ''}>Оценка</Link>
          <Link to="/explain" className={pathname === '/explain' ? 'active' : ''}>SHAP</Link>
        </nav>
      </header>
      <Routes>
        <Route path="/" element={<EstimatePage />} />
        <Route path="/explain" element={<ExplainPage />} />
      </Routes>
    </>
  )
}
