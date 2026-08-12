import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ShapChart from '../components/ShapChart'
import FeatureImportanceChart from '../components/FeatureImportanceChart'
import Loader from '../components/Loader'
import { explain, featureImportance } from '../api/endpoints'
import type { FeatureImportanceItem, ShapResponse } from '../types'
import { useAppCtx } from '../context/AppContext.tsx'

export default function ExplainPage() {
  const navigate = useNavigate()
  const { features, selected } = useAppCtx()
  const model = selected ?? 'XGBoost'

  const [shap, setShap] = useState<ShapResponse | null>(null)
  const [fi, setFi] = useState<FeatureImportanceItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!features) {
      setError('Нет данных квартиры. Вернитесь и выполните оценку.')
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)

    Promise.all([
      explain(model, features),
      featureImportance(model),
    ])
      .then(([s, f]) => { setShap(s); setFi(f.features) })
      .catch(() => setError('Ошибка при получении объяснения'))
      .finally(() => setLoading(false))
  }, [model, features])

  return (
    <div className="container container--single">
      <button
        className="btn-secondary"
        style={{ width: 'auto' }}
        onClick={() => navigate('/')}
      >
        ← Назад к оценке
      </button>

      {loading && <Loader text="Расчёт SHAP…" />}
      {error && <div className="error">{error}</div>}

      {shap && (
        <div className="card">
          <h2 className="card__title">SHAP-разложение оценки</h2>
          <div className="info-block">
            Разложение построено для модели <b>{model}</b>. Зелёные вклады
            повышают цену, красные — понижают. Сумма всех вкладов и базового
            значения равна итоговой оценке.
          </div>
          <ShapChart data={shap} />
        </div>
      )}

      {fi.length > 0 && (
        <div className="card">
          <h2 className="card__title">Глобальная важность признаков · {model}</h2>
          <FeatureImportanceChart data={fi} />
        </div>
      )}
    </div>
  )
}
