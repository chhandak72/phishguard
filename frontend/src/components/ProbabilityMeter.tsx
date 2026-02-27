/**
 * ProbabilityMeter.tsx
 * --------------------
 * Animated progress bar showing phishing probability.
 */

import React from "react";
import type { RiskLevel } from "../types";

interface Props {
  probability: number; // 0.0 – 1.0
  riskLevel: RiskLevel;
}

const RISK_COLOURS: Record<RiskLevel, { bar: string; text: string }> = {
  Low: { bar: "bg-emerald-500", text: "text-emerald-700" },
  Medium: { bar: "bg-amber-400", text: "text-amber-700" },
  High: { bar: "bg-red-500", text: "text-red-700" },
};

const ProbabilityMeter: React.FC<Props> = ({ probability, riskLevel }) => {
  const pct = Math.round(probability * 100);
  const colours = RISK_COLOURS[riskLevel];

  return (
    <div aria-label={`Phishing probability: ${pct}%`}>
      {/* Label row */}
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-medium text-slate-500">
          Phishing probability
        </span>
        <span className={`text-sm font-bold ${colours.text}`}>
          {(probability * 100).toFixed(1)}%
        </span>
      </div>

      {/* Track */}
      <div
        className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {/* Fill – width driven by inline style for smooth animation */}
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${colours.bar}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between mt-1 text-[10px] text-slate-400 select-none">
        <span>0%</span>
        <span>33%</span>
        <span>66%</span>
        <span>100%</span>
      </div>
    </div>
  );
};

export default ProbabilityMeter;
