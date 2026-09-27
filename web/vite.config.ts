import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The build lands inside the Python package so `pip install -e .` users never
// need Node. Fixed file names keep the committed bundle's git history readable.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../dungeon/web/static",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        entryFileNames: "assets/app.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8642", changeOrigin: false } },
  },
});
