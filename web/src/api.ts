import type { Attachment, Message, PrecontextItem, Settings, Usage } from "./types";

export type Health = {
  ok?: boolean;
  model?: string;
  brain_ready?: boolean;
  perception?: { ok?: boolean };
  diarize?: { ok?: boolean };
};

function base(settings: Settings): string {
  return settings.serverUrl.trim().replace(/\/+$/, "");
}

function headers(settings: Settings): Record<string, string> {
  const out: Record<string, string> = { "Content-Type": "application/json" };
  if (settings.apiKey.trim()) out.Authorization = `Bearer ${settings.apiKey.trim()}`;
  return out;
}

export async function fetchHealth(settings: Settings, signal?: AbortSignal): Promise<Health> {
  const res = await fetch(`${base(settings)}/health`, { signal });
  return res.json();
}

export async function fetchModel(settings: Settings): Promise<string | null> {
  try {
    const res = await fetch(`${base(settings)}/v1/models`, { headers: headers(settings) });
    const body = await res.json();
    return body?.data?.[0]?.id ?? null;
  } catch {
    return null;
  }
}

/** The OpenAI content part that carries an attachment. */
function attachmentPart(a: Attachment): Record<string, unknown> {
  if (a.kind === "image") return { type: "image_url", image_url: { url: a.dataUri } };
  if (a.kind === "audio") {
    // The server accepts a data: URI here and keeps its declared type.
    const format = a.mime.split("/")[1]?.replace(/^x-/, "").replace("mpeg", "mp3") || "wav";
    return { type: "input_audio", input_audio: { data: a.dataUri, format } };
  }
  return { type: "file", file: { file_data: a.dataUri, filename: a.name } };
}

/** The conversation as chat completion messages. Assistant turns go back as their prose only. */
export function buildMessages(history: Message[], settings: Settings): Record<string, unknown>[] {
  const system: string[] = [];
  if (settings.task) system.push(`<task>${settings.task}</task>`);
  if (settings.guard.length) system.push(`<guard>${settings.guard.join(", ")}</guard>`);
  if (settings.systemPrompt.trim()) system.push(settings.systemPrompt.trim());

  const out: Record<string, unknown>[] = [];
  if (system.length) out.push({ role: "system", content: system.join("\n\n") });
  for (const m of history) {
    if (m.role === "user") {
      if (!m.attachments.length) {
        out.push({ role: "user", content: m.text });
      } else {
        const parts: Record<string, unknown>[] = [];
        if (m.text.trim()) parts.push({ type: "text", text: m.text });
        parts.push(...m.attachments.map(attachmentPart));
        out.push({ role: "user", content: parts });
      }
    } else if (m.status === "done" && m.text) {
      out.push({ role: "assistant", content: m.text });
    }
  }
  return out;
}

export class ApiError extends Error {}

export function buildBody(model: string, history: Message[], settings: Settings): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model,
    messages: buildMessages(history, settings),
    stream: true,
  };
  if (settings.reasoningEffort) body.reasoning_effort = settings.reasoningEffort;
  const temperature = parseFloat(settings.temperature);
  if (Number.isFinite(temperature)) body.temperature = temperature;
  const maxTokens = parseInt(settings.maxTokens, 10);
  if (Number.isFinite(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;
  if (settings.jsonSchema.trim()) {
    let schema: unknown;
    try {
      schema = JSON.parse(settings.jsonSchema);
    } catch (e) {
      throw new ApiError(`The JSON schema does not parse: ${(e as Error).message}`);
    }
    body.response_format = { type: "json_schema", json_schema: { name: "response", schema } };
  }
  return body;
}

export type StreamUpdate =
  | { type: "content"; text: string }
  | { type: "finish"; reason: string; usage?: Usage };

/**
 * POST a streamed chat completion and yield its deltas.
 *
 * `x-show-additional-info` asks the server to put each tool's result in the stream,
 * wrapped in `<precontext> ... </precontext>` inside `content`; `splitPrecontext`
 * takes those back out.
 */
export async function* streamChat(
  settings: Settings,
  body: Record<string, unknown>,
  signal: AbortSignal,
): AsyncGenerator<StreamUpdate> {
  const res = await fetch(`${base(settings)}/v1/chat/completions`, {
    method: "POST",
    headers: { ...headers(settings), "x-show-additional-info": "true" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const err = await res.json();
      message = err?.error?.message ?? err?.detail ?? message;
    } catch {
      /* not JSON */
    }
    throw new ApiError(message);
  }
  if (!res.body) throw new ApiError("The server sent no response body.");

  // A server can answer a stream request with plain JSON; take it whole.
  if (!(res.headers.get("content-type") ?? "").includes("event-stream")) {
    const whole = await res.json();
    const choice = whole?.choices?.[0];
    const precontext = whole?.precontext as PrecontextItem[] | undefined;
    if (precontext?.length) {
      yield { type: "content", text: `<precontext> ${JSON.stringify(precontext)} </precontext>` };
    }
    if (choice?.message?.content) yield { type: "content", text: choice.message.content };
    yield { type: "finish", reason: choice?.finish_reason ?? "stop", usage: whole?.usage };
    return;
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      if (data === "[DONE]") return;
      const chunk = JSON.parse(data);
      if (chunk.error) throw new ApiError(chunk.error.message ?? "The response failed while streaming.");
      const choice = chunk.choices?.[0];
      const text = choice?.delta?.content;
      if (text) yield { type: "content", text };
      if (choice?.finish_reason) yield { type: "finish", reason: choice.finish_reason, usage: chunk.usage };
    }
  }
}

const PRECONTEXT = /<precontext> ([\s\S]*?) <\/precontext>/g;

/** The prose of a streamed answer, with the precontext blocks lifted out of it. */
export function splitPrecontext(raw: string): { text: string; items: PrecontextItem[] } {
  const items: PrecontextItem[] = [];
  let text = raw.replace(PRECONTEXT, (_, json: string) => {
    try {
      const parsed = JSON.parse(json);
      if (Array.isArray(parsed)) items.push(...parsed);
    } catch {
      /* a malformed block is dropped rather than shown as prose */
    }
    return "";
  });
  // A block still arriving is not prose yet.
  const open = text.indexOf("<precontext>");
  if (open !== -1) text = text.slice(0, open);
  return { text: text.replace(/^\s+/, ""), items };
}
