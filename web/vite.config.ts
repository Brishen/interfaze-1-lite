import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands inside the Python package, where the orchestrator serves it at `/`.
// `npm run dev` serves the UI on :5173 and forwards the API to a running server.
const api = process.env.INTERFAZE_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/interfaze_lite/web",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/v1": { target: api, changeOrigin: true },
      "/health": { target: api, changeOrigin: true },
    },
  },
});
