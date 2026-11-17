import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: path.join(root, "src"),
  publicDir: path.join(root, "public"),
  build: {
    outDir: path.join(root, "dist"),
    emptyOutDir: true,
    target: "es2022",
    sourcemap: true,
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  resolve: {
    alias: {
      "@shared": path.join(root, "src", "shared"),
      "@kernel": path.join(root, "src", "kernel"),
      "@render": path.join(root, "src", "render"),
    },
  },
});
