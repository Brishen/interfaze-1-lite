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

/** A step the server reports while it works (`x-interfaze-progress`). */
export type ServerProgress = { stage: string; [field: string]: unknown };

export type StreamUpdate =
  | { type: "upload"; sent: number; total: number }
  | { type: "uploaded" }
  | { type: "progress"; event: ServerProgress }
  | { type: "content"; text: string }
  | { type: "finish"; reason: string; usage?: Usage };

/** Callback events turned into an async iterator. */
class Channel<T> {
  private items: T[] = [];
  private wake: (() => void) | null = null;
  private closed = false;
  private failure: unknown = null;

  push(item: T) {
    this.items.push(item);
    this.wake?.();
  }

  close(failure: unknown = null) {
    if (this.closed) return;
    this.closed = true;
    this.failure = failure;
    this.wake?.();
  }

  async *drain(): AsyncGenerator<T> {
    for (;;) {
      while (this.items.length) yield this.items.shift()!;
      if (this.closed) {
        if (this.failure) throw this.failure;
        return;
      }
      await new Promise<void>((resolve) => (this.wake = resolve));
      this.wake = null;
    }
  }
}

function errorMessage(status: number, text: string): string {
  try {
    const err = JSON.parse(text);
    return err?.error?.message ?? err?.detail ?? `${status}`;
  } catch {
    return `The server answered ${status}.`;
  }
}

/**
 * POST a streamed chat completion and yield its deltas, with the request's progress.
 *
 * XMLHttpRequest rather than fetch: only it reports how much of a large upload has been
 * sent. `x-show-additional-info` puts each tool's result in the stream inside
 * `<precontext> ... </precontext>` (see `splitPrecontext`); `x-interfaze-progress` adds
 * `: progress {...}` comment lines as the server plans, runs tools and writes.
 */
export async function* streamChat(
  settings: Settings,
  body: Record<string, unknown>,
  signal: AbortSignal,
): AsyncGenerator<StreamUpdate> {
  const channel = new Channel<StreamUpdate>();
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `${base(settings)}/v1/chat/completions`);
  for (const [k, v] of Object.entries(headers(settings))) xhr.setRequestHeader(k, v);
  xhr.setRequestHeader("x-show-additional-info", "true");
  xhr.setRequestHeader("x-interfaze-progress", "true");

  let seen = 0;
  let buffer = "";
  let finished = false;
  const isStream = () => (xhr.getResponseHeader("content-type") ?? "").includes("event-stream");

  const parse = (final: boolean) => {
    buffer += xhr.responseText.slice(seen);
    seen = xhr.responseText.length;
    if (final && buffer.trim()) buffer += "\n\n";
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const event = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data: string[] = [];
      for (const line of event.split("\n")) {
        if (line.startsWith(": progress ")) {
          try {
            channel.push({ type: "progress", event: JSON.parse(line.slice(11)) });
          } catch {
            /* a progress line is a courtesy; a bad one is skipped */
          }
        } else if (line.startsWith("data:")) {
          data.push(line.slice(5).trimStart());
        }
      }
      const payload = data.join("\n");
      if (!payload) continue;
      if (payload === "[DONE]") {
        finished = true;
        continue;
      }
      const chunk = JSON.parse(payload);
      if (chunk.error) throw new ApiError(chunk.error.message ?? "The response failed while streaming.");
      const choice = chunk.choices?.[0];
      const text = choice?.delta?.content;
      if (text) channel.push({ type: "content", text });
      if (choice?.finish_reason) channel.push({ type: "finish", reason: choice.finish_reason, usage: chunk.usage });
    }
  };

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) channel.push({ type: "upload", sent: e.loaded, total: e.total });
  };
  xhr.upload.onload = () => channel.push({ type: "uploaded" });
  xhr.onprogress = () => {
    if (xhr.status >= 400 || !isStream()) return;
    try {
      parse(false);
    } catch (e) {
      channel.close(e);
      xhr.abort();
    }
  };
  xhr.onload = () => {
    try {
      if (xhr.status >= 400) throw new ApiError(errorMessage(xhr.status, xhr.responseText));
      if (isStream()) {
        parse(true);
      } else {
        // A server can answer a stream request with plain JSON; take it whole.
        const whole = JSON.parse(xhr.responseText);
        const choice = whole?.choices?.[0];
        const precontext = whole?.precontext as PrecontextItem[] | undefined;
        if (precontext?.length) {
          channel.push({ type: "content", text: `<precontext> ${JSON.stringify(precontext)} </precontext>` });
        }
        if (choice?.message?.content) channel.push({ type: "content", text: choice.message.content });
        channel.push({ type: "finish", reason: choice?.finish_reason ?? "stop", usage: whole?.usage });
        finished = true;
      }
      channel.close(finished ? null : new ApiError("The response ended before it was complete."));
    } catch (e) {
      channel.close(e);
    }
  };
  xhr.onerror = () => channel.close(new TypeError("network error"));
  xhr.onabort = () => channel.close(new DOMException("Aborted", "AbortError"));
  signal.addEventListener("abort", () => xhr.abort(), { once: true });

  xhr.send(JSON.stringify(body));
  try {
    yield* channel.drain();
  } finally {
    if (xhr.readyState !== XMLHttpRequest.DONE) xhr.abort();
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

/**
 * The model's reasoning, which the server sends inside the content wrapped in
 * `<think> ... </think>`, apart from the answer. `thinking` is true while the block is
 * still open, i.e. the model is still reasoning.
 */
export function splitThinking(text: string): { reasoning: string; answer: string; thinking: boolean } {
  const open = text.indexOf("<think>");
  if (open === -1) return { reasoning: "", answer: text, thinking: false };
  const close = text.indexOf("</think>", open);
  if (close === -1) return { reasoning: text.slice(open + 7).trimStart(), answer: text.slice(0, open), thinking: true };
  return {
    reasoning: text.slice(open + 7, close).trim(),
    answer: (text.slice(0, open) + text.slice(close + 8)).trimStart(),
    thinking: false,
  };
}
