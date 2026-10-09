import { useMemo, useState } from "react";
import { collectShapes } from "../geometry";
import { ImageOverlay } from "./ImageOverlay";

type Chunk = { text?: string; timestamp?: [number, number]; speaker?: string };
type Prediction = { date?: string; timestamp?: string; value?: number };

const TITLES: Record<string, string> = {
  ocr: "Document reading",
  stt: "Transcription",
  speech_to_text: "Transcription",
  object_detection: "Object detection",
  gui_detection: "GUI grounding",
  translate: "Translation",
  forecast: "Forecast",
  text_guardrail_classifier: "Text guardrail",
  image_guardrail_classifier: "Image guardrail",
  answer: "Answer",
};

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

function clock(s: number | undefined): string {
  if (typeof s !== "number" || !Number.isFinite(s)) return "–";
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

/** The clipboard API exists only in secure contexts; plain HTTP falls back to execCommand. */
function copy(text: string) {
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(text).catch(() => {});
    return;
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

/** Whether a result has more to show than its JSON: boxes to draw, a transcript, a series. */
export function hasRichView(result: unknown, image?: string): boolean {
  const record = asRecord(result);
  return (
    (!!image && collectShapes(result).length > 0) ||
    (Array.isArray(record?.chunks) && record!.chunks.length > 0) ||
    (Array.isArray(record?.predictions) && record!.predictions.length > 0)
  );
}

type View = "overlay" | "transcript" | "forecast" | "text" | "json";

/** One tool's result: drawn over the image when it has boxes, tabulated when it is a transcript or series. */
export function ToolResult({ name, result, image }: { name: string; result: unknown; image?: string }) {
  const shapes = useMemo(() => collectShapes(result), [result]);
  const record = asRecord(result);
  const chunks = Array.isArray(record?.chunks) ? (record!.chunks as Chunk[]) : null;
  const predictions = Array.isArray(record?.predictions) ? (record!.predictions as Prediction[]) : null;
  // Text worth reading as text: a translation, or a document's reading.
  const text = [record?.translated_text, record?.extracted_text].find(
    (t): t is string | string[] => (typeof t === "string" && !!t.trim()) || (Array.isArray(t) && t.length > 0),
  );

  const views: View[] = [];
  if (image && shapes.length) views.push("overlay");
  if (chunks?.length) views.push("transcript");
  if (predictions?.length) views.push("forecast");
  if (text) views.push("text");
  views.push("json");
  const [view, setView] = useState<View>(views[0]);
  const json = useMemo(() => JSON.stringify(result, null, 2), [result]);

  return (
    <div className="tool">
      <div className="tool-head">
        <span className="tool-name">{TITLES[name] ?? name}</span>
        <code className="muted">{name}</code>
        {shapes.length > 0 && <span className="pill">{shapes.length} boxes</span>}
        {typeof result === "string" && result.length < 40 && <span className="pill">{result}</span>}
        <div className="tabs">
          {views.map((v) => (
            <button key={v} className={v === view ? "tab active" : "tab"} onClick={() => setView(v)}>
              {v}
            </button>
          ))}
        </div>
      </div>
      <div className="tool-body">
        {view === "overlay" && image && <ImageOverlay src={image} shapes={shapes} />}
        {view === "transcript" && chunks && (
          <table className="table">
            <tbody>
              {chunks.map((c, i) => (
                <tr key={i}>
                  <td className="mono muted nowrap">
                    {clock(c.timestamp?.[0])}–{clock(c.timestamp?.[1])}
                  </td>
                  {chunks.some((x) => x.speaker) && <td className="nowrap">{c.speaker}</td>}
                  <td className="grow">{c.text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {view === "forecast" && predictions && (
          <table className="table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, i) => (
                <tr key={i}>
                  <td className="mono">{p.date ?? p.timestamp}</td>
                  <td className="mono">{typeof p.value === "number" ? p.value.toFixed(2) : String(p.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {view === "text" && text && (
          <div className="json">
            <button className="ghost small copy" onClick={() => copy(Array.isArray(text) ? text.join("\n\n") : text)}>
              Copy
            </button>
            <pre className="plain">{Array.isArray(text) ? text.join("\n\n") : text}</pre>
          </div>
        )}
        {view === "json" && (
          <div className="json">
            <button className="ghost small copy" onClick={() => copy(json)}>
              Copy
            </button>
            <pre>{json.length > 400_000 ? `${json.slice(0, 400_000)}\n… (truncated)` : json}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
