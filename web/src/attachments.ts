import { uid } from "./ids";
import type { Attachment } from "./types";

/** The server's own per-request ceiling is generous; this keeps a browser tab responsive. */
export const MAX_BYTES = 100 * 1024 * 1024;

export const ACCEPT = "image/*,audio/*,video/webm,.pdf,.docx,.csv,.json,.txt";

function readDataUri(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export async function toAttachment(file: File): Promise<Attachment> {
  if (file.size > MAX_BYTES) {
    throw new Error(`${file.name} is ${(file.size / 1048576).toFixed(0)} MB; the limit here is ${MAX_BYTES / 1048576} MB.`);
  }
  const mime = (file.type || "application/octet-stream").split(";")[0];
  const kind = mime.startsWith("image/") ? "image" : mime.startsWith("audio/") ? "audio" : "file";
  return {
    id: uid(),
    name: file.name,
    mime,
    size: file.size,
    dataUri: await readDataUri(file),
    kind,
  };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
