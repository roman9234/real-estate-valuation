import client from './client'
import type {
  ApartmentRequest,
  FeaturesMeta,
  FeatureImportanceResponse,
  PredictionResponse,
  SensitivityRequestBody,
  SensitivityResponse,
  ShapResponse,
} from '../types'

export const getFeatures = () =>
  client.get<FeaturesMeta>('/features').then((r) => r.data)

export const predict = (body: ApartmentRequest) =>
  client.post<PredictionResponse>('/predict', body).then((r) => r.data)

export const explain = (modelName: string, body: ApartmentRequest) =>
  client
    .post<ShapResponse>(`/explain/${encodeURIComponent(modelName)}`, body)
    .then((r) => r.data)

export const featureImportance = (modelName: string) =>
  client
    .get<FeatureImportanceResponse>(
      `/feature-importance/${encodeURIComponent(modelName)}`,
    )
    .then((r) => r.data)

export const sensitivity = (
  modelName: string,
  featureName: string,
  body: SensitivityRequestBody,
) =>
  client
    .post<SensitivityResponse>(
      `/sensitivity/${encodeURIComponent(modelName)}/${encodeURIComponent(featureName)}`,
      body,
    )
    .then((r) => r.data)
