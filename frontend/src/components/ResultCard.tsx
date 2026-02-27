/**
 * ResultCard.tsx
 * --------------
 * Displays the prediction result with:
 * - Label badge (Phishing / Legitimate)
 * - Probability meter
 * - Risk level pill
 * - Top feature reasons list
 * - Collapsible "Explain" details
 */

import React, { useState } from "react";
import type { PredictResponse } from "../types";
import ProbabilityMeter from "./ProbabilityMeter";

interface Props {
  result: PredictResponse;
}

const RISK_STYLES = {
  Low: "badge-low",
  Medium: "badge-medium",
  High: "badge-high",
} as const;

const RISK_ICONS = {
  Low: "🟢",
  Medium: "🟡",
  High: "🔴",
} as const;

function formatFeatureName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(value: string | number | boolean): string {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number")
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

const ResultCard: React.FC<Props> = ({ result }) => {
  const [showExplain, setShowExplain] = useState(false);
  const isPhishing = result.label === "Phishing";

  return (
    <section
      className="card p-6 space-y-5 animate-slide-up"
      aria-label="Prediction result"
      aria-live="polite"
    >
      {/* ── Row 1: label + risk ───────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Label badge */}
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-bold
            ${isPhishing ? "label-phish" : "label-legit"}`}
          aria-label={`Classification: ${result.label}`}
        >
          {isPhishing ? "⚠️" : "✅"} {result.label}
        </span>

        {/* Risk level */}
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold
            ${RISK_STYLES[result.risk_level]}`}
          aria-label={`Risk level: ${result.risk_level}`}
        >
          {RISK_ICONS[result.risk_level]} {result.risk_level} Risk
        </span>

        {/* Processing time */}
        <span className="ml-auto text-xs text-slate-400 tabular-nums">
          {result.processing_time_ms.toFixed(0)} ms
        </span>
      </div>

      {/* ── Row 2: probability meter ──────────────── */}
      <ProbabilityMeter
        probability={result.phishing_probability}
        riskLevel={result.risk_level}
      />

      {/* ── Row 3: summary ───────────────────────── */}
      {result.reason_summary && (
        <p className="text-sm text-slate-600 italic border-l-4 border-slate-200 pl-3">
          {result.reason_summary}
        </p>
      )}

      {/* ── Row 4: top reasons ───────────────────── */}
      {result.top_reasons.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Top contributing signals
          </h3>
          <ul
            className="space-y-2"
            role="list"
            aria-label="Feature contributions"
          >
            {result.top_reasons.map((reason, idx) => (
              <li
                key={idx}
                className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2"
              >
                <div className="min-w-0">
                  <span className="block text-sm font-medium text-slate-700 truncate">
                    {formatFeatureName(reason.feature)}
                  </span>
                  <span className="text-xs text-slate-400">
                    value: {formatValue(reason.value)}
                  </span>
                </div>
                <span
                  className={`ml-4 text-sm font-bold tabular-nums flex-shrink-0
                    ${reason.score_contribution > 0.05 ? "text-red-600" : "text-slate-500"}`}
                  aria-label={`Contribution: ${reason.score_contribution}`}
                >
                  +{reason.score_contribution.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Row 5: Explain toggle ─────────────────── */}
      <div>
        <button
          className="btn-secondary text-xs"
          onClick={() => setShowExplain((v) => !v)}
          aria-expanded={showExplain}
          aria-controls="explain-detail"
        >
          {showExplain ? "▲ Hide details" : "▼ Show full details"}
        </button>

        {showExplain && (
          <div
            id="explain-detail"
            className="mt-3 rounded-lg bg-slate-50 p-4 text-xs text-slate-600 space-y-1 animate-fade-in"
            aria-label="Full prediction details"
          >
            <p>
              <strong>Label:</strong> {result.label}
            </p>
            <p>
              <strong>Phishing probability:</strong>{" "}
              {(result.phishing_probability * 100).toFixed(2)}%
            </p>
            <p>
              <strong>Risk level:</strong> {result.risk_level}
            </p>
            <p>
              <strong>All signals:</strong>
            </p>
            <pre className="mt-1 overflow-x-auto whitespace-pre-wrap break-all">
              {JSON.stringify(result.top_reasons, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </section>
  );
};

export default ResultCard;
