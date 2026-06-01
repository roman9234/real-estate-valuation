// ===== Признаки квартиры (вход) — строго по schemas/request.py =====
export type RenovationType =
  | 'Cosmetic'
  | 'Designer'
  | 'European-style renovation'
  | 'Without renovation'

export interface ApartmentRequest {
  area: number
  rooms: number
  floor: number
  total_floors: number
  minutes_to_metro: number
  is_studio: number // 0 | 1
  metro_station: string
  renovation: RenovationType
}

// ===== Метаданные признаков — по выходу /features (адаптер) =====
export interface NumericFeatureMeta {
  name: string
  dtype: string
  min: number
  max: number
  median: number
  mean: number
  q01: number
  q99: number
  label: string
  input_type: 'number' | 'checkbox'
  step?: number
  unit?: string
  derived: boolean
}

export interface CategoricalFeatureMeta {
  name: string
  _unique: number
  values: string[]
  label: string
  input_type: 'select'
  derived: boolean
}

export interface FeaturesMeta {
  numeric: NumericFeatureMeta[]
  categorical: CategoricalFeatureMeta[]
  target: { name: string; min: number; max: number; median: number; mean: number }
}

// ===== Ответы — строго по schemas/response.py =====
export interface MetricsResponse {
  MAE: number
  RMSE: number
  MAPE: number
  R2: number
  best_params?: Record<string, unknown> | null
}

export interface PredictionItem {
  model_name: string
  price: number
  ci_lower: number
  ci_upper: number
  price_per_sqm: number
  metrics: MetricsResponse
}

export interface PredictionResponse {
  predictions: PredictionItem[]
}

export interface FeatureImportanceItem {
  feature_name: string
  importance: number
}
export interface FeatureImportanceResponse {
  model_name: string
  features: FeatureImportanceItem[]
}

export interface ShapValueItem {
  feature_name: string
  value: number
}
export interface ShapResponse {
  model_name: string
  base_value: number
  prediction: number
  shap_values: ShapValueItem[]
}

export interface SensitivityPoint {
  value: number | string
  price: number
}
export interface SensitivityResponse {
  model_name: string
  feature_name: string
  points: SensitivityPoint[]
}

// Тело sensitivity-запроса
export interface SensitivityRequestBody extends ApartmentRequest {
  range_min?: number
  range_max?: number
  step?: number
}
