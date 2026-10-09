import { useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { formatBytes } from "../attachments";
import type { AssistantMessage, UserMessage } from "../types";
import { Progress } from "./Progress";
import { Reasoning } from "./Reasoning";
import { hasRichView, ToolResult } from "./ToolResult";

export function UserView({ message }: { message: UserMessage }) {
  return (
    <div className="msg user">
      <div className="bubble">
        {message.attachments.length > 0 && (
          <div className="attachments">
            {message.attachments.map((a) =>
              a.kind === "image" ? (
                <img key={a.id} className="thumb" src={a.dataUri} alt={a.name} title={a.name} />
              ) : a.kind === "audio" ? (
                <div key={a.id} className="file-chip audio">
                  <span>{a.name}</span>
                  <audio controls src={a.dataUri} />
                </div>
              ) : (
                <div key={a.id} className="file-chip">
                  <span className="file-icon">{a.name.split(".").pop()?.toUpperCase().slice(0, 4)}</span>
                  <span>{a.name}</span>
                  <span className="muted">{formatBytes(a.size)}</span>
                </div>
              ),
            )}
          </div>
        )}
        {message.text && <div className="user-text">{message.text}</div>}
      </div>
    </div>
  );
}

/** JSON answers (a run task, or a schema) are shown as data, not as prose. */
function parseJson(text: string): unknown | undefined {
  const t = text.trim();
  if (!t.startsWith("{") && !t.startsWith("[")) return undefined;
  try {
    return JSON.parse(t);
  } catch {
    return undefined;
  }
}

function Stats({ message }: { message: AssistantMessage }) {
  const parts: string[] = [];
  if (message.finishedAt) parts.push(`${((message.finishedAt - message.startedAt) / 1000).toFixed(1)} s`);
  if (message.firstTokenAt) parts.push(`first text ${((message.firstTokenAt - message.startedAt) / 1000).toFixed(1)} s`);
  const u = message.usage;
  if (u) {
    parts.push(`${u.prompt_tokens.toLocaleString()} in · ${u.completion_tokens.toLocaleString()} out`);
    const r = u.completion_tokens_details?.reasoning_tokens;
    if (r) parts.push(`${r.toLocaleString()} reasoning`);
  }
  if (message.finishReason && !["stop", "length"].includes(message.finishReason)) parts.push(`finish: ${message.finishReason}`);
  if (message.status === "stopped") parts.push("stopped");
  return parts.length ? <div className="stats">{parts.join(" · ")}</div> : null;
}

export function AssistantView({ message, image }: { message: AssistantMessage; image?: string }) {
  const json = useMemo(() => (message.status === "done" ? parseJson(message.text) : undefined), [message.status, message.text]);
  // A run task answers with `{name, result}`, the same result its tool card already
  // holds. Shown once, as the answer, under the tool's own title.
  const routed = useMemo(() => {
    const r = json as { name?: unknown; result?: unknown } | undefined;
    return r && typeof r === "object" && typeof r.name === "string" && "result" in r ? { name: r.name, result: r.result } : null;
  }, [json]);
  const tools = useMemo(() => {
    if (!routed) return message.precontext;
    const same = JSON.stringify(routed.result);
    return message.precontext.filter((p) => JSON.stringify(p.result) !== same);
  }, [message.precontext, routed]);
  // Open when there is something to look at; a result that is only JSON (a PDF's OCR)
  // starts folded, so it does not push the answer out of view.
  const [toggled, setToggled] = useState<boolean | null>(null);
  const rich = useMemo(() => tools.some((t) => hasRichView(t.result, image)), [tools, image]);
  const open = toggled ?? rich;
  const live = message.status === "streaming";

  return (
    <div className="msg assistant">
      <Progress
        steps={message.steps}
        live={live}
        folded={!!message.text}
        startedAt={message.startedAt}
        finishedAt={message.finishedAt}
        waitingLabel={message.steps.length ? "Working…" : "Sending the request…"}
      />
      {message.reasoning && <Reasoning text={message.reasoning} live={live && !!message.thinking} />}
      {tools.length > 0 && (
        <div className="tools">
          <button className="ghost small" onClick={() => setToggled(!open)}>
            {open ? "▾" : "▸"} {tools.length} tool result{tools.length > 1 ? "s" : ""}
          </button>
          {open &&
            tools.map((item, i) => (
              <ToolResult key={i} name={item.name} result={item.result} image={image} />
            ))}
        </div>
      )}
      {routed ? (
        <ToolResult name={routed.name} result={routed.result} image={image} />
      ) : json !== undefined ? (
        <ToolResult name="answer" result={json} image={image} />
      ) : (
        message.text && (
          <div className={message.status === "streaming" ? "prose streaming" : "prose"}>
            <Markdown remarkPlugins={[remarkGfm]}>{message.text}</Markdown>
          </div>
        )
      )}
      {message.status === "done" && message.finishReason === "length" && (
        <div className="notice">
          The answer reached the model's output limit and was cut off.
          {tools.some((t) => hasRichView(t.result, image)) && " The complete result is in the tool card above."}
        </div>
      )}
      {message.error && <div className="error">{message.error}</div>}
      <Stats message={message} />
    </div>
  );
}
