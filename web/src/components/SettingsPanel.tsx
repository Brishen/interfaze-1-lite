import { GUARD_CODES, TASKS, type Settings } from "../types";

type Props = {
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
  onReset: () => void;
};

const EXAMPLE_SCHEMA = `{
  "type": "object",
  "properties": {
    "vendor": { "type": "string" },
    "total": { "type": "number" },
    "line_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": { "type": "string" },
          "amount": { "type": "number" }
        }
      }
    }
  },
  "required": ["vendor", "total"]
}`;

export function SettingsPanel({ settings, onChange, onReset }: Props) {
  const schemaError = (() => {
    if (!settings.jsonSchema.trim()) return null;
    try {
      JSON.parse(settings.jsonSchema);
      return null;
    } catch (e) {
      return (e as Error).message;
    }
  })();
  const taskAndSchema = !!settings.task && !!settings.jsonSchema.trim();

  return (
    <div className="settings">
      <section>
        <h3>Run</h3>
        <label className="field">
          <span>Task</span>
          <select value={settings.task} onChange={(e) => onChange({ task: e.target.value })}>
            {TASKS.map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
          <small>A task skips planning and returns that capability's raw result.</small>
        </label>
        <label className="field">
          <span>Reasoning effort</span>
          <select
            value={settings.reasoningEffort}
            onChange={(e) => onChange({ reasoningEffort: e.target.value as Settings["reasoningEffort"] })}
          >
            <option value="">Off</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </label>
        <div className="field-row">
          <label className="field">
            <span>Temperature</span>
            <input
              type="number"
              step="0.1"
              min="0"
              max="2"
              placeholder="default"
              value={settings.temperature}
              onChange={(e) => onChange({ temperature: e.target.value })}
            />
          </label>
          <label className="field">
            <span>Max tokens</span>
            <input
              type="number"
              min="1"
              placeholder="default"
              value={settings.maxTokens}
              onChange={(e) => onChange({ maxTokens: e.target.value })}
            />
          </label>
        </div>
        <label className="field">
          <span>System prompt</span>
          <textarea
            rows={3}
            value={settings.systemPrompt}
            placeholder="Optional instructions for every turn"
            onChange={(e) => onChange({ systemPrompt: e.target.value })}
          />
        </label>
      </section>

      <section>
        <h3>
          Structured output
          {!settings.jsonSchema && (
            <button className="ghost small" onClick={() => onChange({ jsonSchema: EXAMPLE_SCHEMA })}>
              Insert example
            </button>
          )}
        </h3>
        <label className="field">
          <span>JSON schema</span>
          <textarea
            className="mono"
            rows={settings.jsonSchema ? 10 : 3}
            value={settings.jsonSchema}
            placeholder="Paste a JSON schema to get the answer as an object that fills it"
            onChange={(e) => onChange({ jsonSchema: e.target.value })}
          />
          {schemaError && <small className="error-text">{schemaError}</small>}
          {taskAndSchema && <small className="error-text">A schema can't be combined with a task.</small>}
        </label>
      </section>

      <section>
        <h3>
          Guardrails
          {settings.guard.length > 0 && (
            <button className="ghost small" onClick={() => onChange({ guard: [] })}>
              Clear
            </button>
          )}
        </h3>
        <small className="muted">Screen each request first; a blocked one is answered with the verdict.</small>
        <div className="chips">
          {GUARD_CODES.map(([code, label]) => {
            const on = settings.guard.includes(code);
            return (
              <button
                key={code}
                className={on ? "chip on" : "chip"}
                title={label}
                onClick={() =>
                  onChange({ guard: on ? settings.guard.filter((c) => c !== code) : [...settings.guard, code] })
                }
              >
                <b>{code}</b> {label}
              </button>
            );
          })}
        </div>
      </section>

      <section>
        <h3>Connection</h3>
        <label className="field">
          <span>Server URL</span>
          <input
            value={settings.serverUrl}
            placeholder="same origin"
            onChange={(e) => onChange({ serverUrl: e.target.value })}
          />
        </label>
        <label className="field">
          <span>API key</span>
          <input
            type="password"
            value={settings.apiKey}
            placeholder="only if the server sets API_KEY"
            autoComplete="off"
            onChange={(e) => onChange({ apiKey: e.target.value })}
          />
          <small>Kept in this browser's local storage.</small>
        </label>
      </section>

      <button className="ghost" onClick={onReset}>
        Reset settings
      </button>
    </div>
  );
}
