/**
 * Header.tsx
 * ----------
 * PhishGuard top navigation / hero header.
 */

import React from "react";

const ShieldIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-8 h-8"
    aria-hidden="true"
  >
    <path
      fillRule="evenodd"
      d="M12 1.5a.75.75 0 0 1 .652.379l1.273 2.214A17.03 17.03 0 0 0 21 6.447V8.25a9.75 9.75 0 0 1-9.75 9.75A9.75 9.75 0 0 1 1.5 8.25V6.447a17.03 17.03 0 0 0 7.075-1.854l1.273-2.214A.75.75 0 0 1 12 1.5Zm0 3.077L11.16 6.22a15.53 15.53 0 0 1-6.41 1.706V8.25a8.25 8.25 0 0 0 8.25 8.25 8.25 8.25 0 0 0 8.25-8.25V7.926a15.53 15.53 0 0 1-6.41-1.706L12 4.577Z"
      clipRule="evenodd"
    />
    <path d="M10.5 11.25a.75.75 0 0 0 0 1.5h3a.75.75 0 0 0 0-1.5h-3Z" />
  </svg>
);

const Header: React.FC = () => {
  return (
    <header
      className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm"
      role="banner"
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        {/* Logo + brand */}
        <div className="flex items-center gap-3">
          <span className="text-blue-600" aria-hidden="true">
            <ShieldIcon />
          </span>
          <div>
            <h1 className="text-xl font-bold text-slate-900 leading-tight tracking-tight">
              PhishGuard
            </h1>
            <p className="text-xs text-slate-500 leading-none">
              ML-powered phishing detector
            </p>
          </div>
        </div>

        {/* Status pill */}
        <span
          className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium
                     text-emerald-700 bg-emerald-50 border border-emerald-200
                     rounded-full px-3 py-1"
          aria-label="Model status: active"
        >
          <span
            className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"
            aria-hidden="true"
          />
          Model active
        </span>
      </div>
    </header>
  );
};

export default Header;
