import { useEffect, useRef, useState } from "react";
import { ACCEPT, formatBytes, toAttachment } from "../attachments";
import type { Attachment } from "../types";

type Props = {
  busy: boolean;
  onSend: (text: string, attachments: Attachment[]) => void;
  onStop: () => void;
};

export function Composer({ busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [recording, setRecording] = useState<MediaRecorder | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const add = async (files: Iterable<File>) => {
    setError(null);
    for (const file of files) {
      try {
        const a = await toAttachment(file);
        setAttachments((prev) => [...prev, a]);
      } catch (e) {
        setError((e as Error).message);
      }
    }
  };

  const send = () => {
    if (busy || (!text.trim() && !attachments.length)) return;
    onSend(text, attachments);
    setText("");
    setAttachments([]);
  };

  const toggleRecording = async () => {
    if (recording) {
      recording.stop();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Recording needs HTTPS or localhost; attach an audio file instead.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const parts: Blob[] = [];
      recorder.ondataavailable = (e) => parts.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(null);
        const type = (recorder.mimeType || "audio/webm").split(";")[0];
        const ext = type.split("/")[1] ?? "webm";
        const stamp = new Date().toTimeString().slice(0, 8).replace(/:/g, "");
        add([new File(parts, `recording-${stamp}.${ext}`, { type })]);
      };
      recorder.start();
      setRecording(recorder);
    } catch (e) {
      setError(`Microphone unavailable: ${(e as Error).message}`);
    }
  };

  return (
    <div
      className={dragging ? "composer dragging" : "composer"}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        add(e.dataTransfer.files);
      }}
    >
      {attachments.length > 0 && (
        <div className="pending">
          {attachments.map((a) => (
            <div key={a.id} className="pending-item">
              {a.kind === "image" ? <img src={a.dataUri} alt="" /> : <span className="file-icon">{a.name.split(".").pop()?.toUpperCase().slice(0, 4)}</span>}
              <span className="pending-name" title={a.name}>
                {a.name}
              </span>
              <span className="muted">{formatBytes(a.size)}</span>
              <button className="ghost small" title="Remove" onClick={() => setAttachments((p) => p.filter((x) => x.id !== a.id))}>
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
      {error && <div className="error small">{error}</div>}
      <div className="composer-row">
        <button className="icon" title="Attach images, audio, PDFs, Word files, CSV or JSON" onClick={() => fileInput.current?.click()}>
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>
        <button className={recording ? "icon recording" : "icon"} title={recording ? "Stop recording" : "Record audio"} onClick={toggleRecording}>
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M19 10v1a7 7 0 0 1-14 0v-1M12 18v4" />
          </svg>
        </button>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept={ACCEPT}
          hidden
          onChange={(e) => {
            if (e.target.files) add(e.target.files);
            e.target.value = "";
          }}
        />
        <textarea
          ref={textarea}
          rows={1}
          value={text}
          placeholder="Ask anything, or drop in files"
          onChange={(e) => setText(e.target.value)}
          onPaste={(e) => {
            const files = [...e.clipboardData.files];
            if (files.length) {
              e.preventDefault();
              add(files);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        {busy ? (
          <button className="primary" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="primary" onClick={send} disabled={!text.trim() && !attachments.length}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}
