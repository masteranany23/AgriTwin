export type ApiHealth = { status: string; service?: string; version?: string; database?: string };
export type Field = {
  field_id: string; farm_id: string; name: string; latitude: number; longitude: number;
  area_ha?: number | null; elevation_m?: number | null; description?: string | null;
  boundary_geojson?: Record<string, unknown> | null; simulation_count: number;
  created_at?: string | null; updated_at?: string | null;
};
export type FieldList = { total: number; limit: number; offset: number; items: Field[] };
export type DailyState = Record<string, string | number | null>;
export type Simulation = {
  simulation_id: string; field_id?: string | null; run_type: string; status: string;
  model_name: string; crop: string; variety: string; latitude: number; longitude: number;
  sowing_date: string; harvest_date?: string | null; yield_kg_ha?: number | null;
  peak_lai?: number | null; total_days?: number | null; created_at?: string | null;
};
export type SimulationDetail = Simulation & {
  model_version: string; error_message?: string | null; use_real_weather: boolean;
  use_real_soil: boolean; daily_states: DailyState[]; summary_payload?: Record<string, unknown> | null;
  metrics_payload?: Record<string, unknown> | null; warnings?: string[] | null;
};
export type SimulationList = { total: number; limit: number; offset: number; items: Simulation[] };
export type TimeSeriesPoint = { date: string; open_loop?: number | null; assimilated?: number | null; observation?: number | null };
export type TimeSeries = Record<string, TimeSeriesPoint[]>;
export type AssimilationStatus = {
  assimilation_run_id: string; latest_assimilation_run: string; status: string; ensemble_size: number;
  total_cycles: number; executed_cycles: number; skipped_cycles: number;
  latest_cycle_date?: string | null; observations_assimilated: number;
};
export type Cycle = {
  cycle_date: string; variables_updated: string[]; observation_vector: Record<string, number | null>;
  prior_state: Record<string, number | null>; posterior_state: Record<string, number | null>;
  innovation: Record<string, number | null>; quality_score?: number | null; cycle_number: number;
};
export type YieldPoint = { date: string; predicted_yield_kg_ha?: number | null };
export type Diagnostics = {
  simulation_id: string; assimilation_run_id?: string | null; total_cycles: number; executed_cycles: number;
  total_valid_obs: number; total_rejected_obs: number; avg_state_update_magnitude: Record<string, number | null>;
  avg_innovation: Record<string, number | null>; mean_prior_spread: Record<string, number | null>;
  mean_posterior_spread: Record<string, number | null>; cycles: Array<Record<string, unknown>>;
};
export type Observation = {
  observation_id?: string; id?: string; field_id?: string | null; simulation_run_id?: string | null;
  timestamp: string; variable_name: string; units: string; value: number; uncertainty: number;
  source: string; provider_name: string; latitude?: number | null; longitude?: number | null;
  quality_score?: number | null; cloud_cover?: number | null; status?: string | null;
};
export type ObservationList = { total: number; limit: number; offset: number; items: Observation[] };
export type SatelliteScene = {
  acquisition_date: string; cloud_cover: number; ndvi?: number | null; osavi?: number | null;
  seli?: number | null; estimated_lai?: number | null; quality_score: number; metadata: Record<string, unknown>;
};

const base = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...init,
    headers: { ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...init?.headers },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json() as { detail?: string | Array<{ msg?: string }> };
      message = typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail) ? body.detail.map((item) => item.msg).filter(Boolean).join(', ') : message;
    } catch { /* Preserve the useful status message when the server is not JSON. */ }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });
const params = (values: Record<string, string | number | undefined | null>) => {
  const query = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => value !== undefined && value !== null && value !== '' && query.set(key, String(value)));
  return query.toString() ? `?${query.toString()}` : '';
};

export const api = {
  health: () => request<ApiHealth>('/health'),
  fields: (filters: Record<string, string | number | undefined> = {}) => request<FieldList>(`/fields${params({ limit: 50, ...filters })}`),
  field: (id: string) => request<Field>(`/fields/${id}`),
  createField: (body: Record<string, unknown>) => request<Field>('/fields', json(body)),
  deleteField: (id: string) => request<void>(`/fields/${id}`, { method: 'DELETE' }),
  simulations: (filters: Record<string, string | number | undefined> = {}) => request<SimulationList>(`/simulations${params({ limit: 50, ...filters })}`),
  simulation: (id: string) => request<SimulationDetail>(`/simulations/${id}`),
  crops: () => request<Record<string, string[]>>('/simulate/crops'),
  simulate: (body: Record<string, unknown>) => request<SimulationDetail & { simulation_id?: string }>(`/simulate`, json(body)),
  assimilationStatus: (id: string) => request<AssimilationStatus>(`/assimilation/status/${id}`),
  assimilationHistory: (id: string) => request<Cycle[]>(`/assimilation/${id}/history`),
  assimilationTimeseries: (id: string) => request<TimeSeries>(`/assimilation/${id}/timeseries`),
  yieldEvolution: (id: string) => request<YieldPoint[]>(`/assimilation/${id}/yield-evolution`),
  assimilationDiagnostics: (id: string) => request<Diagnostics>(`/assimilation/${id}/diagnostics`),
  forecast: (id: string) => request<Record<string, unknown>>(`/assimilation/${id}/forecast`),
  runAssimilation: (body: { simulation_id: string; field_id: string; ensemble_size: number }) => request<Record<string, unknown>>('/assimilation/run', json(body)),
  observations: (filters: Record<string, string | number | undefined> = {}) => request<ObservationList>(`/observations${params({ limit: 100, ...filters })}`),
  latestObservations: (fieldId: string) => request<Observation[]>(`/observations/latest${params({ field_id: fieldId })}`),
  byVariable: (variable: string) => request<Observation[]>(`/observations/by-variable${params({ variable_name: variable })}`),
  createObservation: (body: Record<string, unknown>) => request<Observation>('/observations', json(body)),
  satelliteLai: (query: Record<string, string | number>) => request<SatelliteScene[]>(`/satellite/lai${params(query)}`),
  fusion: (operation: string, body: unknown) => request<Record<string, unknown>>(`/fusion/${operation}`, json(body)),
  scenario: (type: string, body: unknown, queryParams?: Record<string, string | number | undefined>) => request<Record<string, unknown>>(`/scenarios/${type}${params(queryParams || {})}`, json(body)),
  benchmark: (body: unknown) => request<Record<string, unknown>>('/benchmark/evaluate', json(body)),
  benchmarkDiagnostics: (id: string) => request<Record<string, unknown>>(`/benchmark/diagnostics/${id}`),
  scout: (fieldId: string, images: File[], notes: string) => {
    const body = new FormData();
    images.forEach((image) => body.append('images', image));
    body.append('session_notes', notes);
    return request<Record<string, unknown>>(`/fields/${fieldId}/scout-session`, { method: 'POST', body });
  },
};

export const apiBase = base;