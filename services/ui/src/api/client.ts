/**
 * Axios client + 5 typed API functions.
 *
 * `baseURL: "/api"` so the same code works in:
 *   - dev: Vite proxy `/api -> http://localhost:8000` (vite.config.ts).
 *   - prod: nginx in the UI container proxies `/api -> http://gateway:8000`.
 *
 * The gateway (port 8000) is the single source of truth for the wire
 * format. See `types/api.ts` for the contract.
 */

import axios, { AxiosError, type AxiosInstance } from "axios";
import type {
  DatasetId,
  DatasetsResponse,
  GatewayHealthResponse,
  LogClickRequest,
  RagRequest,
  RagResponse,
  RefineRequest,
  RefineResponse,
  SearchRequest,
  SearchResponse,
} from "../types/api";

export const api: AxiosInstance = axios.create({
  baseURL: "/api",
  timeout: 30_000,
});

// Response interceptor: log failures with status + url so dev console
// shows the cause even when the body is empty (e.g. 204 from /log/click).
api.interceptors.response.use(
  (r) => r,
  (err: AxiosError) => {
    // eslint-disable-next-line no-console
    console.error(
      "[api]",
      err.response?.status,
      err.config?.url,
      err.message
    );
    return Promise.reject(err);
  }
);

/**
 * `POST /search` → unified search endpoint.
 * Routes by `representation` field to the right downstream.
 */
export function search(req: SearchRequest): Promise<SearchResponse> {
  return api.post<SearchResponse>("/search", req).then((r) => r.data);
}

/**
 * `POST /refine` → refinement service (port 8004). Returns the
 * cleaned / spell-corrected / synonym-expanded / personalized
 * query + per-token weights.
 */
export function refine(req: RefineRequest): Promise<RefineResponse> {
  return api.post<RefineResponse>("/refine", req).then((r) => r.data);
}

/**
 * `POST /rag/answer` → currently a 501 stub from the gateway. Kept
 * here so Phase 8 can swap in the real implementation without
 * changing call sites.
 */
export function ragAnswer(req: RagRequest): Promise<RagResponse> {
  return api.post<RagResponse>("/rag/answer", req).then((r) => r.data);
}

/**
 * `GET /datasets` → the canonical list of corpus ids. Mirrors
 * `shared.ir_common.schemas.DATASET_IDS`.
 */
export function listDatasets(): Promise<DatasetId[]> {
  return api
    .get<DatasetsResponse>("/datasets")
    .then((r) => r.data.datasets);
}

/**
 * `POST /log/click` → refinement service `/log/click`. The gateway
 * returns 204 No Content; we return `void` so the caller doesn't
 * accidentally try to read a body.
 */
export async function logClick(payload: LogClickRequest): Promise<void> {
  await api.post("/log/click", payload);
}

/**
 * `GET /health` (gateway root, not /api/health). Returns 200 with
 * a `services` map. Useful for the "is the gateway up?" banner.
 */
export function health(): Promise<GatewayHealthResponse> {
  // The gateway exposes /health on its root (not under /api), so we
  // hit the bare URL with a leading slash.
  return api.get<GatewayHealthResponse>("/health").then((r) => r.data);
}

/**
 * Helper: extract a user-friendly error message from an AxiosError.
 * The gateway's 502/503 body is a `GatewayErrorResponse` (nested
 * inside `detail`), so we walk the structure to surface `detail`.
 */
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const body = err.response?.data as
      | { detail?: string | { detail?: string } }
      | undefined;
    if (status === 501) {
      return "RAG service is not yet available (Phase 8).";
    }
    if (body?.detail) {
      if (typeof body.detail === "string") return body.detail;
      if (typeof body.detail === "object" && "detail" in body.detail) {
        return String(body.detail.detail);
      }
    }
    if (status === undefined) {
      return "Network error: is the gateway running on :8000?";
    }
    return `Request failed (HTTP ${status}).`;
  }
  if (err instanceof Error) return err.message;
  return "Unknown error.";
}
