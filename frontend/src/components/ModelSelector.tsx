// ModelSelector.tsx — переключает отображаемую модель из уже полученного ответа
import type { PredictionItem } from '../types'

interface Props {
  items: PredictionItem[]
  selected: string
  onSelect: (name: string) => void
}

export default function ModelSelector({ items, selected, onSelect }: Props) {
  return (
    <div className="card">
      <h2 className="card__title">Модель машинного обучения</h2>
      <table className="metrics-table">
        <thead>
          <tr>
            <th></th>
            <th>Модель</th>
            <th>MAE</th>
            <th>MAPE</th>
            <th>R²</th>
          </tr>
        </thead>
        <tbody>
          {items.map((m) => (
            <tr
              key={m.model_name}
              className={m.model_name === selected ? 'is-selected' : ''}
              onClick={() => onSelect(m.model_name)}
            >
              <td>
                <input
                  type="radio"
                  checked={m.model_name === selected}
                  onChange={() => onSelect(m.model_name)}
                />
              </td>
              <td>
                {m.model_name}
                {m.model_name === 'XGBoost' && (
                  <span className="badge">рекомендуется</span>
                )}
              </td>
              <td>{new Intl.NumberFormat('ru-RU').format(Math.round(m.metrics.MAE))}</td>
              <td>{m.metrics.MAPE.toFixed(1)}%</td>
              <td>{m.metrics.R2.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
