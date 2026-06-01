// PredictionResult.tsx
import type { PredictionItem } from '../types'

const fmt = (n: number) =>
  new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 }).format(n)

export default function PredictionResult({ item }: { item: PredictionItem }) {
  return (
    <div className="result">
      <div className="result__model">{item.model_name}</div>
      <div className="result__price">{fmt(item.price)} ₽</div>
      <div className="result__ci">
        Доверительный интервал: {fmt(item.ci_lower)} — {fmt(item.ci_upper)} ₽
      </div>
      <div className="result__sqm">
        Цена за м²: {fmt(item.price_per_sqm)} ₽
      </div>
      <div className="result__metrics">
        MAPE {item.metrics.MAPE.toFixed(2)}% · R² {item.metrics.R2.toFixed(3)} ·
        MAE {fmt(item.metrics.MAE)} ₽
      </div>
    </div>
  )
}
