/**
 * Footer.tsx
 * ----------
 * Site footer with exact author attribution.
 */

import React from "react";

const Footer: React.FC = () => {
  return (
    <footer
      className="mt-auto border-t border-slate-200 bg-white py-6"
      role="contentinfo"
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 text-center">
        <p className="text-sm text-slate-500">
          Built by{" "}
          <span className="font-medium text-slate-700">
            Aayush Paudel 2260308
          </span>
          ,{" "}
          <span className="font-medium text-slate-700">
            Chhandak Mukherjee 2260357
          </span>{" "}
          and{" "}
          <span className="font-medium text-slate-700">
            Arnav Makhija 2260344
          </span>
        </p>
        <p className="mt-1 text-xs text-slate-400">
          PhishGuard v1.0.0 · ML-powered phishing email classifier
        </p>
      </div>
    </footer>
  );
};

export default Footer;
