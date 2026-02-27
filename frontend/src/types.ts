/**
 * types.ts
 * --------
 * Shared TypeScript interfaces for PhishGuard.
 */

export interface FeatureContribution {
  feature: string;
  value: string | number | boolean;
  score_contribution: number;
}

export interface PredictResponse {
  label: 'Phishing' | 'Legitimate';
  phishing_probability: number;   // 0.0 – 1.0
  risk_level: 'Low' | 'Medium' | 'High';
  top_reasons: FeatureContribution[];
  reason_summary: string;
  processing_time_ms: number;
}

export interface PredictRequest {
  subject?: string;
  body?: string;
  email_raw?: string;
}

export type RiskLevel = 'Low' | 'Medium' | 'High';
