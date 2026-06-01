import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ApartmentForm from '../components/ApartmentForm'
import ModelSelector from '../components/ModelSelector'
import PredictionResult from '../components/PredictionResult'
import SensitivityPanel from '../components/SensitivityPanel'
import Loader from '../components/Loader'
import { predict } from '../api/endpoints'
import { useFeatures } from '../hooks/useFeatures'
import type { ApartmentRequest } from '../types'
import {useAppCtx} from "../context/AppContext.tsx";

export default function EstimatePage() {
  const { meta, loading: metaLoading, error: metaError } = useFeatures()
  const navigate = useNavigate()

  const { features, predictions, selected, setFeatures, setPredictions, setSelected } = useAppCtx()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sensFeature, setSensFeature] = useState<string | null>(null)

  if (metaLoading) return <Loader text="Загрузка каталога признаков…" />
  if (metaError || !meta) return <div className="error">{metaError}</div>

  const onSubmit = async (data: ApartmentRequest) => {
    setLoading(true); setError(null)
    try {
      const res = await predict(data)
      setPredictions(res.predictions)
      setFeatures(data)
    } catch {
      setError('Ошибка при оценке. Проверьте параметры.')
    } finally {
      setLoading(false)
    }
  }

  const current = predictions.find((p) => p.model_name === selected)
  const numericMeta = meta.numeric.find((n) => n.name === sensFeature) ?? null
  const catMeta = meta.categorical.find((c) => c.name === sensFeature) ?? null

  return (
    <>
      <div className="container">
        <div>
          <ApartmentForm meta={meta} loading={loading} onSubmit={onSubmit} />
        </div>

        <div>
          <div className="card">
            <h2 className="card__title">Результат оценки</h2>
            {error && <div className="error">{error}</div>}
            {!predictions.length && !error && (
              <div className="placeholder">
                Заполните параметры слева и нажмите «Оценить стоимость»
              </div>
            )}
            {current && <PredictionResult item={current} />}
          </div>

          {predictions.length > 0 && (
            <>
              <div style={{ height: 24 }} />
              <ModelSelector items={predictions} selected={selected} onSelect={setSelected} />

              <div style={{ height: 24 }} />
              <div className="card">
                <h2 className="card__title">Анализ чувствительности</h2>
                <p style={{ fontSize: 13, color: '#4a5868' }}>
                  Выберите признак, чтобы построить график зависимости цены:
                </p>
                {[...meta.numeric, ...meta.categorical].map((f) => (
                  <button key={f.name} className="btn-icon"
                    onClick={() => setSensFeature(f.name)}>
                    📈 {f.label}
                  </button>
                ))}
              </div>

              <div style={{ height: 24 }} />
              <button className="btn-primary"
                onClick={() =>
                  navigate('/explain', { state: { features, model: selected } })
                }>
                Открыть SHAP-объяснение ({selected})
              </button>
            </>
          )}
        </div>
      </div>

      {sensFeature && features && (
        <SensitivityPanel
          modelName={selected}
          baseFeatures={features}
          numeric={numericMeta}
          categorical={catMeta}
          onClose={() => setSensFeature(null)}
        />
      )}
    </>
  )
}
