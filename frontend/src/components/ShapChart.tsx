import { useMemo, useState } from 'react'
import {
  Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import type { ShapResponse } from '../types'

const fmt = (n: number) => new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 }).format(n)
type SortMode = 'alpha' | 'value'

export default function ShapChart({ data }: { data: ShapResponse }) {
  const [sort, setSort] = useState<SortMode>('value')

  // Готовим waterfall: каждый бар = [start, end], цвет по знаку вклада.
  const chartData = useMemo(() => {
    const items = [...data.shap_values]
    if (sort === 'alpha') {
      items.sort((a, b) => a.feature_name.localeCompare(b.feature_name, 'ru'))
    } else {
      items.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    }
    let acc = data.base_value
    return items.map((it) => {
      const start = acc
      const end = acc + it.value
      acc = end
      return {
        name: it.feature_name,
        range: [start, end] as [number, number],
        value: it.value,
        positive: it.value >= 0,
      }
    })
  }, [data, sort])

  return (
    <div>
      <div className="sort-toggle">
        <button
          className={sort === 'value' ? 'active' : ''}
          onClick={() => setSort('value')}
        >
          По вкладу (₽)
        </button>
        <button
          className={sort === 'alpha' ? 'active' : ''}
          onClick={() => setSort('alpha')}
        >
          По алфавиту
        </button>
      </div>

      <ResponsiveContainer width="100%" height={40 * chartData.length + 60}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30 }}>
          <XAxis type="number" tickFormatter={(v) => fmt(v)} />
          <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 12 }} />
          <Tooltip
            formatter={(_v, _n, p) => {
              const row = p.payload as { value: number }
              return [`${row.value >= 0 ? '+' : ''}${fmt(row.value)} ₽`, 'Вклад']
            }}
          />
          <Bar dataKey="range">
            {chartData.map((d, i) => (
              <Cell key={i} fill={d.positive ? '#2ecc71' : '#e74c3c'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="result__metrics">
        <span>Базовое значение: {fmt(data.base_value)} ₽</span>
        <span>Итоговая оценка: {fmt(data.prediction)} ₽</span>
      </div>
    </div>
  )
}
