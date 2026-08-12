import { createContext, useContext, useState, type ReactNode } from 'react'
import type { ApartmentRequest, PredictionItem } from '../types'

interface AppState {
  features: ApartmentRequest | null
  predictions: PredictionItem[]
  selected: string
}
interface AppContextValue extends AppState {
  setFeatures: (f: ApartmentRequest | null) => void
  setPredictions: (p: PredictionItem[]) => void
  setSelected: (m: string) => void
}

const AppCtx = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [features, setFeatures] = useState<ApartmentRequest | null>(null)
  const [predictions, setPredictions] = useState<PredictionItem[]>([])
  const [selected, setSelected] = useState('XGBoost')
  return (
    <AppCtx.Provider value={{ features, predictions, selected, setFeatures, setPredictions, setSelected }}>
      {children}
    </AppCtx.Provider>
  )
}

export function useAppCtx() {
  const ctx = useContext(AppCtx)
  if (!ctx) throw new Error('useAppCtx must be inside AppProvider')
  return ctx
}
