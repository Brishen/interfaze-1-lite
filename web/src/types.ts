export type Attachment = {
  id: string;
  name: string;
  mime: string;
  size: number;
  /** A data: URI of the whole file. */
  dataUri: string;
  kind: "image" | "audio" | "file";
};

/** One internal tool's result, as the server reports it in `precontext`. */
export type PrecontextItem = { name: string; result: unknown };

export type Usage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  completion_tokens_details?: { reasoning_tokens?: number };
};

export type UserMessage = {
  id: string;
  role: "user";
  text: string;
  attachments: Attachment[];
};

/** One stage of working out an answer, as the progress list shows it. */
export type Step = {
  key: string;
  label: string;
  detail?: string;
  state: "running" | "done" | "failed";
  startedAt: number;
  endedAt?: number;
  /** Bytes sent and to send, for the upload. */
  sent?: number;
  total?: number;
  /** Runs alongside its siblings rather than after them (a tool call). */
  parallel?: boolean;
};

export type AssistantMessage = {
  id: string;
  role: "assistant";
  text: string;
  precontext: PrecontextItem[];
  steps: Step[];
  /** The model's reasoning, when it was asked to reason. */
  reasoning?: string;
  /** Still reasoning: the reasoning is open and the answer has not begun. */
  thinking?: boolean;
  status: "streaming" | "done" | "error" | "stopped";
  error?: string;
  finishReason?: string;
  usage?: Usage;
  startedAt: number;
  firstTokenAt?: number;
  finishedAt?: number;
};

export type Message = UserMessage | AssistantMessage;

export const TASKS = [
  { id: "", label: "Auto (the model picks tools)" },
  { id: "ocr", label: "OCR" },
  { id: "speech_to_text", label: "Speech to text" },
  { id: "object_detection", label: "Object detection" },
  { id: "gui_detection", label: "GUI detection" },
  { id: "translate", label: "Translate" },
  { id: "forecast", label: "Forecast" },
] as const;

export const GUARD_CODES: [string, string][] = [
  ["S1", "Violent crimes"],
  ["S2", "Non-violent crimes"],
  ["S3", "Sex-related crimes"],
  ["S4", "Child sexual exploitation"],
  ["S5", "Defamation"],
  ["S6", "Specialized advice"],
  ["S7", "Privacy"],
  ["S8", "Intellectual property"],
  ["S9", "Indiscriminate weapons"],
  ["S10", "Hate"],
  ["S11", "Suicide & self-harm"],
  ["S12", "Sexual content"],
  ["S13", "Elections"],
  ["S14", "Code interpreter abuse"],
  ["S1_IMAGE", "Image: gore"],
  ["S12_IMAGE", "Image: nudity"],
  ["S15_IMAGE", "Image: NSFW"],
];

export type Settings = {
  serverUrl: string;
  apiKey: string;
  systemPrompt: string;
  task: string;
  reasoningEffort: "" | "low" | "medium" | "high";
  temperature: string;
  maxTokens: string;
  jsonSchema: string;
  guard: string[];
};

export const DEFAULT_SETTINGS: Settings = {
  serverUrl: "",
  apiKey: "",
  systemPrompt: "",
  task: "",
  reasoningEffort: "",
  temperature: "",
  maxTokens: "",
  jsonSchema: "",
  guard: [],
};
