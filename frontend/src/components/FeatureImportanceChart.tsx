import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { FeatureImportanceItem } from '../types'

export default function FeatureImportanceChart({ data }: { data: FeatureImportanceItem[] }) {
  const top = [...data].sort((a, b) => b.importance - a.importance).slice(0, 15)
  return (
    <ResponsiveContainer width="100%" height={40 * top.length + 40}>
      <BarChart data={top} layout="vertical" margin={{ left: 20, right: 20 }}>
        <XAxis type="number" />
        <YAxis type="category" dataKey="feature_name" width={170} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v: number) => v.toFixed(4)} />
        <Bar dataKey="importance" fill="#4f86c6" />
      </BarChart>
    </ResponsiveContainer>
  )
}
