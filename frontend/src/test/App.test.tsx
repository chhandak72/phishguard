/**
 * App.test.tsx
 * ------------
 * Basic React Testing Library tests for the PhishGuard UI.
 *
 * Run with:  npm test  (uses vitest in jsdom environment)
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";

// Mock the API module so tests don't make real HTTP calls
vi.mock("../services/api", () => ({
  predictEmail: vi.fn(),
  checkHealth: vi
    .fn()
    .mockResolvedValue({ status: "ok", model_version: "1.0.0-test" }),
  getApiErrorMessage: (err: unknown) =>
    err instanceof Error ? err.message : "API error",
}));

import { predictEmail } from "../services/api";
const mockPredictEmail = vi.mocked(predictEmail);

const PHISHING_RESPONSE = {
  label: "Phishing" as const,
  phishing_probability: 0.87,
  risk_level: "High" as const,
  top_reasons: [
    { feature: "has_url", value: 1, score_contribution: 0.13 },
    { feature: "has_suspicious_words", value: 1, score_contribution: 0.087 },
    { feature: "num_exclamations", value: 3, score_contribution: 0.043 },
  ],
  reason_summary: "Contains URLs; suspicious words detected.",
  processing_time_ms: 42.5,
};

const LEGIT_RESPONSE = {
  label: "Legitimate" as const,
  phishing_probability: 0.12,
  risk_level: "Low" as const,
  top_reasons: [],
  reason_summary: "No strong phishing signals detected.",
  processing_time_ms: 38.1,
};

describe("PhishGuard App", () => {
  beforeEach(() => {
    mockPredictEmail.mockReset();
  });

  // ── Rendering ────────────────────────────────────────────
  it("renders the PhishGuard heading", () => {
    render(<App />);
    expect(screen.getByText("PhishGuard")).toBeInTheDocument();
  });

  it("renders the subject input", () => {
    render(<App />);
    expect(screen.getByLabelText(/email subject/i)).toBeInTheDocument();
  });

  it("renders the body textarea", () => {
    render(<App />);
    expect(screen.getByLabelText(/email body/i)).toBeInTheDocument();
  });

  it("renders the Predict button", () => {
    render(<App />);
    expect(
      screen.getByRole("button", { name: /predict/i }),
    ).toBeInTheDocument();
  });

  // ── Footer / authors ─────────────────────────────────────
  it("renders the exact author attribution", () => {
    render(<App />);
    expect(screen.getByText(/Aayush Paudel 2260308/)).toBeInTheDocument();
    expect(screen.getByText(/Chhandak Mukherjee 2260357/)).toBeInTheDocument();
    expect(screen.getByText(/Arnav Makhija 2260344/)).toBeInTheDocument();
  });

  // ── Validation ───────────────────────────────────────────
  it("shows error when both subject and body are empty", async () => {
    render(<App />);
    const btn = screen.getByRole("button", { name: /predict/i });
    await userEvent.click(btn);
    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(mockPredictEmail).not.toHaveBeenCalled();
  });

  // ── Happy path: phishing result ──────────────────────────
  it("calls predictEmail and shows phishing result", async () => {
    mockPredictEmail.mockResolvedValueOnce(PHISHING_RESPONSE);
    render(<App />);

    const subjectInput = screen.getByLabelText(/email subject/i);
    const bodyInput = screen.getByLabelText(/email body/i);
    const btn = screen.getByRole("button", { name: /predict/i });

    await userEvent.type(subjectInput, "Urgent: verify now");
    await userEvent.type(
      bodyInput,
      "Click http://evil.xyz to claim your prize!",
    );
    await userEvent.click(btn);

    await waitFor(() => {
      expect(mockPredictEmail).toHaveBeenCalledTimes(1);
    });

    // Label badge
    expect(await screen.findByText(/phishing/i)).toBeInTheDocument();
    // Risk level
    expect(await screen.findByText(/high risk/i)).toBeInTheDocument();
  });

  // ── Happy path: legitimate result ────────────────────────
  it("shows legitimate result when probability is low", async () => {
    mockPredictEmail.mockResolvedValueOnce(LEGIT_RESPONSE);
    render(<App />);

    await userEvent.type(
      screen.getByLabelText(/email subject/i),
      "Team meeting",
    );
    await userEvent.type(
      screen.getByLabelText(/email body/i),
      "See you Thursday.",
    );
    await userEvent.click(screen.getByRole("button", { name: /predict/i }));

    expect(await screen.findByText(/legitimate/i)).toBeInTheDocument();
    expect(await screen.findByText(/low risk/i)).toBeInTheDocument();
  });

  // ── Error handling ───────────────────────────────────────
  it("displays API error message on failure", async () => {
    mockPredictEmail.mockRejectedValueOnce(
      new Error("Cannot connect to backend."),
    );
    render(<App />);

    await userEvent.type(
      screen.getByLabelText(/email body/i),
      "Some email body",
    );
    await userEvent.click(screen.getByRole("button", { name: /predict/i }));

    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(await screen.findByText(/cannot connect/i)).toBeInTheDocument();
  });

  // ── Clear button ─────────────────────────────────────────
  it("clear button resets the form and result", async () => {
    mockPredictEmail.mockResolvedValueOnce(PHISHING_RESPONSE);
    render(<App />);

    await userEvent.type(screen.getByLabelText(/email subject/i), "Test");
    await userEvent.type(screen.getByLabelText(/email body/i), "Body text");
    await userEvent.click(screen.getByRole("button", { name: /predict/i }));
    await screen.findByText(/phishing/i);

    // Clear
    await userEvent.click(screen.getByRole("button", { name: /clear/i }));

    expect(screen.getByLabelText(/email subject/i)).toHaveValue("");
    expect(screen.getByLabelText(/email body/i)).toHaveValue("");
    expect(screen.queryByText(/phishing/i)).not.toBeInTheDocument();
  });

  // ── Keyboard / accessibility ─────────────────────────────
  it("Predict button is keyboard-focusable", () => {
    render(<App />);
    const btn = screen.getByRole("button", { name: /predict/i });
    btn.focus();
    expect(document.activeElement).toBe(btn);
  });

  // ── Loading state ────────────────────────────────────────
  it("shows loading state while request is pending", async () => {
    // Never resolves — simulate slow backend
    mockPredictEmail.mockReturnValueOnce(new Promise(() => {}));
    render(<App />);

    await userEvent.type(screen.getByLabelText(/email body/i), "Loading test");
    fireEvent.click(screen.getByRole("button", { name: /predict/i }));

    expect(await screen.findByText(/analysing/i)).toBeInTheDocument();
  });
});
