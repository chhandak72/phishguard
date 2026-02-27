/**
 * api.ts
 * ------
 * Axios-based API client for the PhishGuard backend.
 *
 * The base URL is read from the VITE_API_URL environment variable.
 * In development, Vite's proxy rewrites /api → http://localhost:8000,
 * so the default base URL of '/api' works out of the box.
 *
 * In production (Docker), set VITE_API_URL=http://localhost:8000 (or
 * your deployed backend URL) at build time.
 */

import axios, { AxiosError } from 'axios';
import type { PredictRequest, PredictResponse } from '../types';

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api';

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,  // 30-second timeout for model inference
  headers: {
    'Content-Type': 'application/json',
  },
});

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/**
 * POST /predict
 * Classify an email and return risk level + explanations.
 */
export async function predictEmail(payload: PredictRequest): Promise<PredictResponse> {
  const resp = await apiClient.post<PredictResponse>('/predict', payload);
  return resp.data;
}

/**
 * GET /health
 * Liveness check – returns model version and status.
 */
export async function checkHealth(): Promise<{ status: string; model_version?: string }> {
  const resp = await apiClient.get<{ status: string; model_version?: string }>('/health');
  return resp.data;
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

export function getApiErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d) => d.msg ?? String(d)).join('; ');
    if (err.response?.status === 503) return 'Model is not loaded. Please train the model first.';
    if (err.response?.status === 400) return 'Invalid input. Please provide email subject or body.';
    if (err.code === 'ECONNREFUSED' || err.code === 'ERR_NETWORK') {
      return 'Cannot connect to backend. Make sure the server is running at ' + BASE_URL;
    }
    return err.message || 'Unknown API error.';
  }
  if (err instanceof Error) return err.message;
  return 'An unexpected error occurred.';
}
