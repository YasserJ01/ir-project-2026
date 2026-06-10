import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: {
      // SSE streaming endpoint — disable compression so http-proxy
      // does not buffer the entire response before forwarding.
      "/api/rag/answer/stream": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.removeHeader("accept-encoding");
          });
          proxy.on("proxyRes", (proxyRes) => {
            delete proxyRes.headers["content-length"];
          });
        },
      },

      // String-key /api does NOT strip the prefix in Vite 5.4.x,
      // so we forward the path unchanged to the gateway.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
