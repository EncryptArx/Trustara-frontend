export type DeepfakeResult = {
  label: 'DEEPFAKE' | 'REAL' | string,
  confidence: number,
  media_type: string,
  timestamp: string,
  model: string,
  geo_tag: string,
  preview_url: string
}
