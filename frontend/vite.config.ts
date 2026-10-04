import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built assets are served by FastAPI at /static/* (api.py mounts
// frontend/dist there), so every emitted URL must be rooted there.
export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
