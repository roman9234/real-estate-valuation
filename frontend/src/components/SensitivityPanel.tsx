import { useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { sensitivity } from '../api/endpoints'
import type {
  ApartmentRequest, CategoricalFeatureMeta, NumericFeatureMeta, SensitivityResponse,
} from '../types'

const fmt = (n: number) => new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 }).format(n)

interface Props {
  modelName: string
  baseFeatures: ApartmentRequest
  numeric: NumericFeatureMeta | null
  categorical: CategoricalFeatureMeta | null
  onClose: () => void
}

export default function SensitivityPanel({
  modelName, baseFeatures, numeric, categorical, onClose,
}: Props) {
  const feature = numeric ?? categorical!
  const isNumeric = !!numeric

  // числовой ввод
  const [rangeMin, setRangeMin] = useState<number>(numeric ? numeric.q01 : 0)
  const [rangeMax, setRangeMax] = useState<number>(numeric ? numeric.q99 : 0)
  const [step, setStep] = useState<number>(numeric?.step ?? 1)

  // категориальный выбор
  const [selected, setSelected] = useState<string[]>(categorical?.values ?? [])
  const [search, setSearch] = useState('')

  const [result, setResult] = useState<SensitivityResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const filtered = useMemo(
    () =>
      (categorical?.values ?? []).filter((v) =>
        v.toLowerCase().includes(search.toLowerCase()),
      ),
    [categorical, search],
  )

  const toggle = (v: string) =>
    setSelected((s) => (s.includes(v) ? s.filter((x) => x !== v) : [...s, v]))

  const build = async () => {
    setError(null)
    if (isNumeric) {
      if (rangeMin >= rangeMax) return setError('Начало должно быть меньше конца')
      if (step <= 0) return setError('Шаг должен быть положительным')
    } else if (selected.length === 0) {
      return setError('Выберите хотя бы одну категорию')
    }
    setLoading(true)
    try {
      const res = await sensitivity(modelName, feature.name, {
        ...baseFeatures,
        // фронт отправляет "как надо"; backend пока игнорирует (см. TODO бэка)
        range_min: isNumeric ? rangeMin : undefined,
        range_max: isNumeric ? rangeMax : undefined,
        step: isNumeric ? step : undefined,
        categories: isNumeric ? undefined : selected,
      } as never)
      // для категориального — оставляем только выбранные категории
      const points = isNumeric
        ? res.points
        : res.points.filter((p) => selected.includes(String(p.value)))
      setResult({ ...res, points })
    } catch (e: unknown) {
      setError('Ошибка при построении графика')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__head">
          <h2 className="card__title" style={{ border: 'none', margin: 0 }}>
            Анализ чувствительности · {feature.label}
          </h2>
          <button className="modal__close" onClick={onClose}>×</button>
        </div>

        <div className="info-block">
          Модель: <b>{modelName}</b>. Остальные параметры зафиксированы значениями
          из формы.
        </div>

        {isNumeric ? (
          <div className="row" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
            <div className="field">
              <label>От</label>
              <input type="number" value={rangeMin}
                onChange={(e) => setRangeMin(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>До</label>
              <input type="number" value={rangeMax}
                onChange={(e) => setRangeMax(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Шаг</label>
              <input type="number" value={step}
                onChange={(e) => setStep(Number(e.target.value))} />
            </div>
          </div>
        ) : (
          <>
            <div className="field cat-search">
              <input placeholder="Поиск…" value={search}
                onChange={(e) => setSearch(e.target.value)} />
            </div>
            <div className="cat-actions">
              <button className="btn-secondary"
                onClick={() => setSelected(categorical!.values)}>
                Выбрать все
              </button>
              <button className="btn-secondary" onClick={() => setSelected([])}>
                Снять все
              </button>
            </div>
            <div className="cat-list">
              {filtered.map((v) => (
                <label className="cat-item" key={v}>
                  <input type="checkbox" checked={selected.includes(v)}
                    onChange={() => toggle(v)} />
                  {v}
                </label>
              ))}
            </div>
          </>
        )}

        {error && <div className="error">{error}</div>}

        <button className="btn-primary" onClick={build} disabled={loading}>
          {loading ? 'Построение…' : 'Построить график'}
        </button>

        {result && (
          <div style={{ marginTop: 20 }}>
            <ResponsiveContainer width="100%" height={360}>
              {isNumeric ? (
                <LineChart data={result.points} margin={{ left: 30, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="value" tickFormatter={(v) => String(v)} />
                  <YAxis tickFormatter={(v) => fmt(v)} width={80} />
                  <Tooltip formatter={(v: number) => `${fmt(v)} ₽`} />
                  <Line type="monotone" dataKey="price" stroke="#4f86c6" dot={false} />
                </LineChart>
              ) : (
                <BarChart data={result.points} margin={{ left: 30, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="value" tick={{ fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => fmt(v)} width={80} />
                  <Tooltip formatter={(v: number) => `${fmt(v)} ₽`} />
                  <Bar dataKey="price" fill="#4f86c6" />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}
