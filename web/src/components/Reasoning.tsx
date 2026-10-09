import { useEffect, useRef } from "react";

/** The model's reasoning: streamed in view while it thinks, folded away once it answers. */
export function Reasoning({ text, live }: { text: string; live: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (live && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [text, live]);

  const words = text.split(/\s+/).filter(Boolean).length;
  if (live) {
    return (
      <div className="reasoning live">
        <div className="reasoning-head">
          <span className="spinner" /> Reasoning · {words} words
        </div>
        <div className="reasoning-text" ref={box}>
          {text}
        </div>
      </div>
    );
  }
  return (
    <details className="reasoning">
      <summary>Reasoning · {words} words</summary>
      <div className="reasoning-text">{text}</div>
    </details>
  );
}
