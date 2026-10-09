import type { ServerProgress } from "./api";
import { formatBytes } from "./attachments";
import type { Step } from "./types";

/** What each tool is doing, in words. */
export const TOOL_ACTIONS: Record<string, string> = {
  ocr: "Reading the document",
  stt: "Transcribing the audio",
  object_detection: "Finding objects",
  gui_detection: "Locating interface elements",
  translate: "Translating",
  forecast: "Forecasting",
};

/** Uploads smaller than this finish too fast to be worth a row. */
const SHOW_UPLOAD_FROM = 512 * 1024;

function closeSequential(steps: Step[], now: number): Step[] {
  return steps.map((s) => (s.state === "running" && !s.parallel ? { ...s, state: "done", endedAt: now } : s));
}

function start(steps: Step[], step: Omit<Step, "state" | "startedAt">, now: number): Step[] {
  return [...closeSequential(steps, now), { ...step, state: "running", startedAt: now }];
}

export function onUpload(steps: Step[], sent: number, total: number, now: number): Step[] {
  if (total < SHOW_UPLOAD_FROM) return steps;
  const existing = steps.find((s) => s.key === "upload");
  if (!existing) {
    return [...steps, { key: "upload", label: `Uploading ${formatBytes(total)}`, state: "running", startedAt: now, sent, total }];
  }
  return steps.map((s) => (s.key === "upload" ? { ...s, sent, total } : s));
}

export function onUploaded(steps: Step[], now: number): Step[] {
  return steps.map((s) => (s.key === "upload" && s.state === "running" ? { ...s, state: "done", endedAt: now, sent: s.total } : s));
}

export function onProgress(steps: Step[], event: ServerProgress, now: number): Step[] {
  const n = steps.length;
  switch (event.stage) {
    case "received":
      return onUploaded(steps, now);
    case "guard":
      return start(steps, { key: `guard-${n}`, label: "Checking the request against guardrails", detail: (event.codes as string[] | undefined)?.join(", ") }, now);
    case "model": {
      const step = Number(event.step) || 1;
      const label = event.after_tools ? "Reading the results" : step === 1 ? "Planning" : "Thinking it over";
      return start(steps, { key: `model-${n}`, label }, now);
    }
    case "tool_start":
      return [
        ...closeSequential(steps, now),
        {
          key: `tool-${event.id}`,
          label: TOOL_ACTIONS[String(event.tool)] ?? `Running ${event.tool}`,
          detail: (event.detail as string) || undefined,
          state: "running",
          startedAt: now,
          parallel: true,
        },
      ];
    case "tool_end":
      return steps.map((s) =>
        s.key === `tool-${event.id}` ? { ...s, state: event.ok === false ? "failed" : "done", endedAt: now } : s,
      );
    case "cut_off": {
      // The model's tool call ran past its output limit; the server has asked it to retry.
      let i = steps.length - 1;
      while (i >= 0 && !(steps[i].key.startsWith("model-") && steps[i].state === "running")) i--;
      if (i === -1) return steps;
      return steps.map((s, j) =>
        j === i ? { ...s, state: "failed", endedAt: now, detail: "its tool call was too long to finish; retrying" } : s,
      );
    }
    case "structuring":
      return start(steps, { key: `schema-${n}`, label: event.retry ? "Filling the schema again" : "Filling the schema" }, now);
    case "reasoning":
      return start(steps, { key: `reason-${n}`, label: "Reasoning" }, now);
    case "writing":
      return start(steps, { key: `write-${n}`, label: "Writing the answer" }, now);
    default:
      return steps;
  }
}

/** Answer text has started to arrive: whatever ran before it is now writing the answer. */
export function onText(steps: Step[], now: number): Step[] {
  const last = steps[steps.length - 1];
  if (!last || last.state !== "running" || last.label === "Writing the answer") return steps;
  if (last.key.startsWith("model-")) return [...steps.slice(0, -1), { ...last, label: "Writing the answer" }];
  // Reasoning is over once the answer begins; the answer gets its own row and time.
  if (last.key.startsWith("reason-")) return start(steps, { key: `write-${steps.length}`, label: "Writing the answer" }, now);
  return steps;
}

export function finish(steps: Step[], now: number, failed = false): Step[] {
  return steps.map((s) => (s.state === "running" ? { ...s, state: failed ? "failed" : "done", endedAt: now } : s));
}
