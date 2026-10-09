/**
 * A unique id for list keys. `crypto.randomUUID` exists only in secure contexts (HTTPS or
 * localhost), and the UI is often opened over plain HTTP on a LAN address.
 */
let counter = 0;
export function uid(): string {
  return `${Date.now().toString(36)}-${(counter++).toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
