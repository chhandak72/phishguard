/**
 * App.tsx
 * -------
 * PhishGuard – main single-page application.
 *
 * Layout
 * ──────
 * <Header />
 * <main>
 *   EmailForm  (subject + body inputs, Predict button)
 *   ResultCard (label, probability, risk, reasons)
 *   ErrorBanner
 * </main>
 * <Footer />
 */

import React, { useState, useRef } from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import ResultCard from "./components/ResultCard";
import { predictEmail, getApiErrorMessage } from "./services/api";
import type { PredictResponse } from "./types";

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------
const Spinner: React.FC = () => (
  <svg
    className="animate-spin h-5 w-5 text-white"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
    />
  </svg>
);

// ---------------------------------------------------------------------------
// Error banner
// ---------------------------------------------------------------------------
interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

const ErrorBanner: React.FC<ErrorBannerProps> = ({ message, onDismiss }) => (
  <div
    role="alert"
    aria-live="assertive"
    className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 animate-fade-in"
  >
    <span className="text-base flex-shrink-0" aria-hidden="true">
      ⚠️
    </span>
    <p className="flex-1">{message}</p>
    <button
      onClick={onDismiss}
      className="flex-shrink-0 text-red-400 hover:text-red-600 focus:outline-none focus:ring-2 focus:ring-red-300 rounded"
      aria-label="Dismiss error"
    >
      ✕
    </button>
  </div>
);

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
const App: React.FC = () => {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resultRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!subject.trim() && !body.trim()) {
      setError("Please enter an email subject or body before predicting.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const resp = await predictEmail({ subject, body });
      setResult(resp);
      // Scroll result into view on mobile
      setTimeout(() => {
        resultRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });
      }, 100);
    } catch (err) {
      setError(getApiErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSubject("");
    setBody("");
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header />

      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 py-8 space-y-6">
        {/* ── Page heading ─────────────────────────── */}
        <div className="text-center sm:text-left">
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            Email Analysis
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Paste an email subject and body to check for phishing signals.
          </p>
        </div>

        {/* ── Input form ───────────────────────────── */}
        <form
          onSubmit={handleSubmit}
          className="card p-6 space-y-4"
          aria-label="Email input form"
          noValidate
        >
          {/* Subject */}
          <div>
            <label
              htmlFor="email-subject"
              className="block text-sm font-medium text-slate-700 mb-1.5"
            >
              Subject
            </label>
            <input
              id="email-subject"
              type="text"
              className="input-field"
              placeholder="e.g. Urgent: Verify your account"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              aria-label="Email subject"
              maxLength={500}
              disabled={loading}
            />
          </div>

          {/* Body */}
          <div>
            <label
              htmlFor="email-body"
              className="block text-sm font-medium text-slate-700 mb-1.5"
            >
              Body
              <span className="ml-1 font-normal text-slate-400">
                (paste full email text)
              </span>
            </label>
            <textarea
              id="email-body"
              className="input-field resize-y min-h-[160px]"
              placeholder="Paste the email body here…"
              value={body}
              onChange={(e) => setBody(e.target.value)}
              aria-label="Email body"
              aria-multiline="true"
              rows={7}
              maxLength={50_000}
              disabled={loading}
            />
            <p className="mt-1 text-right text-xs text-slate-400 tabular-nums">
              {body.length.toLocaleString()} / 50,000 chars
            </p>
          </div>

          {/* Actions */}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              type="submit"
              className="btn-primary"
              disabled={loading}
              aria-busy={loading}
            >
              {loading ? (
                <>
                  <Spinner />
                  Analysing…
                </>
              ) : (
                <>
                  <span aria-hidden="true">🔍</span>
                  Predict
                </>
              )}
            </button>

            {(subject || body || result) && (
              <button
                type="button"
                className="btn-secondary"
                onClick={handleClear}
                disabled={loading}
                aria-label="Clear form and results"
              >
                Clear
              </button>
            )}
          </div>
        </form>

        {/* ── Error ────────────────────────────────── */}
        {error && (
          <ErrorBanner message={error} onDismiss={() => setError(null)} />
        )}

        {/* ── Result ───────────────────────────────── */}
        {result && (
          <div ref={resultRef}>
            <ResultCard result={result} />
          </div>
        )}

        {/* ── Sample emails ────────────────────────── */}
        {!result && !loading && (
          <details className="card">
            <summary className="cursor-pointer px-6 py-4 text-sm font-medium text-slate-600 hover:text-slate-900 select-none">
              📋 Try a sample email
            </summary>
            <div className="px-6 pb-4 space-y-3">
              <SampleButton
                label="Phishing example"
                subject="Urgent: Your account has been compromised"
                body={
                  "Dear Customer,\n\nWe have detected unusual activity on your account. " +
                  "Please click the link below to verify your identity immediately or your account " +
                  "will be suspended within 24 hours.\n\nhttp://192.168.1.1/verify-account\n\n" +
                  "Confirm your password and credit card details to restore access.\n\nBanking Team"
                }
                onSelect={(s, b) => {
                  setSubject(s);
                  setBody(b);
                }}
              />
              <SampleButton
                label="Legitimate example"
                subject="Team meeting rescheduled to Thursday"
                body={
                  "Hi everyone,\n\nJust a quick note to let you know the weekly team sync has " +
                  "been moved from Wednesday to Thursday at 2 PM. Same Zoom link as usual.\n\n" +
                  "Please update your calendars.\n\nThanks,\nSarah"
                }
                onSelect={(s, b) => {
                  setSubject(s);
                  setBody(b);
                }}
              />
            </div>
          </details>
        )}
      </main>

      <Footer />
    </div>
  );
};

// ---------------------------------------------------------------------------
// Sample email loader button
// ---------------------------------------------------------------------------
interface SampleButtonProps {
  label: string;
  subject: string;
  body: string;
  onSelect: (subject: string, body: string) => void;
}

const SampleButton: React.FC<SampleButtonProps> = ({
  label,
  subject,
  body,
  onSelect,
}) => (
  <button
    type="button"
    className="btn-secondary w-full sm:w-auto text-left"
    onClick={() => onSelect(subject, body)}
    aria-label={`Load sample: ${label}`}
  >
    {label}
  </button>
);

export default App;
