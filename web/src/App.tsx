import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, buildBody, fetchHealth, fetchModel, splitPrecontext, splitThinking, streamChat, type Health } from "./api";
import { Composer } from "./components/Composer";
import { AssistantView, UserView } from "./components/MessageView";
import { SettingsPanel } from "./components/SettingsPanel";
import { uid } from "./ids";
import { finish, onProgress, onText, onUpload, onUploaded } from "./steps";
import { DEFAULT_SETTINGS, TASKS, type AssistantMessage, type Attachment, type Message, type Settings } from "./types";

const SETTINGS_KEY = "interfaze-lite.settings";

function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

const EXAMPLES = [
  "Read this invoice and give me the total and due date.",
  "Transcribe this recording and split it by speaker.",
  "Find every car and bicycle in this photo.",
  "Where is the search box on this screenshot?",
  "Translate “Where is the train station?” into Japanese and Swahili.",
];

function StatusDot({ ok, label }: { ok: boolean | undefined; label: string }) {
  return (
    <span className="status" title={`${label}: ${ok === undefined ? "unknown" : ok ? "ready" : "down"}`}>
      <span className={ok ? "led ok" : ok === false ? "led bad" : "led"} />
      {label}
    </span>
  );
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState<Health | null | "error">(null);
  const [model, setModel] = useState("interfaze-lite");
  const [showSettings, setShowSettings] = useState(() => window.innerWidth > 1100);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      /* private mode */
    }
  }, [settings]);

  useEffect(() => {
    let live = true;
    const ctrl = new AbortController();
    const poll = () =>
      fetchHealth(settings, ctrl.signal)
        .then((h) => live && setHealth(h))
        .catch(() => live && setHealth("error"));
    poll();
    fetchModel(settings).then((id) => live && id && setModel(id));
    const timer = setInterval(poll, 15000);
    return () => {
      live = false;
      ctrl.abort();
      clearInterval(timer);
    };
  }, [settings.serverUrl, settings.apiKey]);

  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current && messages.length) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const update = useCallback((id: string, patch: Partial<AssistantMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id && m.role === "assistant" ? { ...m, ...patch } : m)));
  }, []);

  const send = async (text: string, attachments: Attachment[]) => {
    const user: Message = { id: uid(), role: "user", text, attachments };
    const reply: AssistantMessage = {
      id: uid(),
      role: "assistant",
      text: "",
      precontext: [],
      steps: [],
      status: "streaming",
      startedAt: Date.now(),
    };
    const history = [...messages, user];
    setMessages([...history, reply]);
    setBusy(true);
    stick.current = true;

    const ctrl = new AbortController();
    abort.current = ctrl;
    let raw = "";
    let firstTokenAt: number | undefined;
    let steps = reply.steps;
    try {
      const body = buildBody(model, history, settings);
      for await (const event of streamChat(settings, body, ctrl.signal)) {
        const now = Date.now();
        switch (event.type) {
          case "upload":
            steps = onUpload(steps, event.sent, event.total, now);
            update(reply.id, { steps });
            break;
          case "uploaded":
            steps = onUploaded(steps, now);
            update(reply.id, { steps });
            break;
          case "progress":
            steps = onProgress(steps, event.event, now);
            update(reply.id, { steps });
            break;
          case "content": {
            raw += event.text;
            const { text: content, items } = splitPrecontext(raw);
            const { reasoning, answer: prose, thinking } = splitThinking(content);
            if ((prose || reasoning) && !firstTokenAt) firstTokenAt = now;
            if (prose) steps = onText(steps, now);
            update(reply.id, { text: prose, reasoning, thinking, precontext: items, firstTokenAt, steps });
            break;
          }
          case "finish":
            update(reply.id, { finishReason: event.reason, usage: event.usage });
            break;
        }
      }
      const now = Date.now();
      update(reply.id, { status: "done", finishedAt: now, steps: finish(steps, now) });
    } catch (e) {
      const now = Date.now();
      if (ctrl.signal.aborted) {
        update(reply.id, { status: "stopped", finishedAt: now, steps: finish(steps, now, true) });
      } else {
        const message =
          e instanceof ApiError
            ? e.message
            : `Could not reach the server (${(e as Error).message}). Is it running, and is the server URL right?`;
        update(reply.id, { status: "error", error: message, finishedAt: now, steps: finish(steps, now, true) });
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
  };

  /** The most recent image the user sent at or before this turn, for drawing boxes on. */
  const imageFor = (index: number): string | undefined => {
    for (let i = index; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "user") {
        const img = m.attachments.find((a) => a.kind === "image");
        if (img) return img.dataUri;
      }
    }
    return undefined;
  };

  const ready = health && health !== "error" ? health : null;
  const taskLabel = TASKS.find((t) => t.id === settings.task)?.label;

  return (
    <div className={showSettings ? "app with-settings" : "app"}>
      <header className="topbar">
        <div className="brand">
          <span className="logo" />
          <span>Interfaze 1 Lite</span>
        </div>
        <div className="statuses">
          {health === "error" ? (
            <StatusDot ok={false} label="server unreachable" />
          ) : (
            <>
              <StatusDot ok={ready?.brain_ready} label="reasoning core" />
              <StatusDot ok={ready ? ready.perception?.ok !== false : undefined} label="perception" />
              <StatusDot ok={ready ? ready.diarize?.ok !== false : undefined} label="diarization" />
            </>
          )}
        </div>
        <div className="actions">
          <button className="ghost" onClick={() => setMessages([])} disabled={busy || !messages.length}>
            New chat
          </button>
          <button className={showSettings ? "ghost active" : "ghost"} onClick={() => setShowSettings((s) => !s)}>
            Settings
          </button>
        </div>
      </header>

      <main className="chat">
        <div
          className="messages"
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
        >
          <div className="column">
            {messages.length === 0 ? (
              <div className="empty">
                <h1>What should we read, hear or find?</h1>
                <p className="muted">
                  Attach images, PDFs, Word files or audio, and ask in plain language. The model picks its own
                  specialists — OCR, speech, detection, grounding, translation, forecasting — and shows you what each one
                  returned.
                </p>
                <div className="examples">
                  {EXAMPLES.map((e) => (
                    <button key={e} className="example" onClick={() => send(e, [])} disabled={busy}>
                      {e}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) =>
                m.role === "user" ? (
                  <UserView key={m.id} message={m} />
                ) : (
                  <AssistantView key={m.id} message={m} image={imageFor(i)} />
                ),
              )
            )}
          </div>
        </div>
        <div className="column composer-wrap">
          {(settings.task || settings.jsonSchema.trim() || settings.guard.length > 0 || settings.reasoningEffort) && (
            <div className="modes">
              {settings.task && <span className="pill">task: {taskLabel}</span>}
              {settings.jsonSchema.trim() && <span className="pill">JSON schema</span>}
              {settings.guard.length > 0 && <span className="pill">guard: {settings.guard.join(", ")}</span>}
              {settings.reasoningEffort && <span className="pill">reasoning: {settings.reasoningEffort}</span>}
            </div>
          )}
          <Composer busy={busy} onSend={send} onStop={() => abort.current?.abort()} />
        </div>
      </main>

      {showSettings && (
        <aside className="sidebar">
          <SettingsPanel
            settings={settings}
            onChange={(patch) => setSettings((s) => ({ ...s, ...patch }))}
            onReset={() => setSettings(DEFAULT_SETTINGS)}
          />
        </aside>
      )}
    </div>
  );
}
