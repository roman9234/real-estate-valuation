import { useEffect, useState } from 'react'
import { getFeatures } from '../api/endpoints'
import type { FeaturesMeta } from '../types'

export function useFeatures() {
  const [meta, setMeta] = useState<FeaturesMeta | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getFeatures()
      .then(setMeta)
      .catch(() => setError('Не удалось загрузить каталог признаков'))
      .finally(() => setLoading(false))
  }, [])

  return { meta, error, loading }
}
