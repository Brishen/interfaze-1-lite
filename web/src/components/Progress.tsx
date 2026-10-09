import { useEffect, useState } from "react";
import type { Step } from "../types";

/** The current time, refreshed while `live`, so running timers tick. */
function useNow(live: boolean): number {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(timer);
  }, [live]);
  return now;
}

function seconds(ms: number): string {
  return ms < 10_000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms / 1000)} s`;
}

function Icon({ state }: { state: Step["state"] }) {
  if (state === "running") return <span className="spinner" aria-label="running" />;
  if (state === "failed") return <span className="step-icon failed">✕</span>;
  return <span className="step-icon done">✓</span>;
}

function StepRow({ step, now }: { step: Step; now: number }) {
  // `now` ticks every 100 ms, so a step that has just started can be ahead of it.
  const elapsed = Math.max(0, (step.endedAt ?? now) - step.startedAt);
  const uploading = step.total !== undefined && step.state === "running";
  return (
    <li className={`step ${step.state}`}>
      <Icon state={step.state} />
      <span className="step-label">{step.label}</span>
      {step.detail && <span className="step-detail">{step.detail}</span>}
      {uploading && (
        <span className="bar">
          <span style={{ width: `${Math.round(((step.sent ?? 0) / (step.total || 1)) * 100)}%` }} />
        </span>
      )}
      <span className="step-time">{seconds(elapsed)}</span>
    </li>
  );
}

type Props = {
  steps: Step[];
  /** The request is still running. */
  live: boolean;
  /** The answer has started to arrive, so the list folds to a line above it. */
  folded: boolean;
  startedAt: number;
  finishedAt?: number;
  waitingLabel: string;
};

/**
 * What the request is doing: a live list while it works, folded to one line once the
 * answer arrives. With no steps reported (an older server), a timer still shows that
 * it is alive.
 */
export function Progress({ steps, live, folded, startedAt, finishedAt, waitingLabel }: Props) {
  const now = useNow(live);
  const total = Math.max(0, (finishedAt ?? now) - startedAt);

  if (live && !folded) {
    const busy = steps.some((s) => s.state === "running");
    return (
      <div className="progress">
        <ul className="steps">
          {steps.map((s) => (
            <StepRow key={s.key} step={s} now={now} />
          ))}
          {!busy && (
            <li className="step running">
              <span className="spinner" />
              <span className="step-label muted">{waitingLabel}</span>
              <span className="step-time">{seconds(total)}</span>
            </li>
          )}
        </ul>
      </div>
    );
  }

  if (!steps.length) return null;
  return (
    <details className="progress done">
      <summary>
        {live && <span className="spinner" />}
        {steps.length} step{steps.length > 1 ? "s" : ""} · {seconds(total)}
        {steps.some((s) => s.state === "failed") && <span className="error-text"> · a step failed</span>}
      </summary>
      <ul className="steps">
        {steps.map((s) => (
          <StepRow key={s.key} step={s} now={now} />
        ))}
      </ul>
    </details>
  );
}
