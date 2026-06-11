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
  ClusterSearchRequest,
  ClusterSearchResponse,
  DatasetId,
  DatasetsResponse,
  DocResponse,
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
 * `POST /cluster/{dataset_id}/search` → clustering service (port 8006).
 * Applies Mini-Batch K-Means cluster boost to search results.
 */
export function clusterSearch(
  req: ClusterSearchRequest,
): Promise<ClusterSearchResponse> {
  return api
    .post<ClusterSearchResponse>(`/cluster/${req.dataset_id}/search`, req)
    .then((r) => r.data);
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
  return api.post<RagResponse>("/rag/answer", req, { timeout: 180_000 }).then((r) => r.data);
}

export interface RagStreamCallbacks {
  onToken: (token: string) => void;
  onDone: (answer: string, sources: string[], latencyMs: number, refinedQuery: string | null, citations?: Record<string, string>) => void;
  onError: (err: string) => void;
  onStage?: (stage: string, data: Record<string, unknown>) => void;
}

export async function ragAnswerStream(
  req: RagRequest,
  callbacks: RagStreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/rag/answer/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!resp.ok) {
    let detail = "RAG stream request failed";
    try {
      const body = await resp.json();
      if (body?.detail) detail = String(body.detail);
    } catch { /* ignore */ }
    callbacks.onError(detail);
    return;
  }
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop()!;
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6);
      if (raw === "[DONE]") return;
      try {
        const payload = JSON.parse(raw);
        if (payload.stage) {
          callbacks.onStage?.(payload.stage, payload);
        } else if (payload.done) {
          callbacks.onDone(
            payload.answer ?? "",
            payload.source_doc_ids ?? [],
            payload.latency_ms ?? 0,
            payload.refined_query ?? null,
            payload.citations ?? undefined,
          );
        } else if (payload.override) {
          callbacks.onDone(
            payload.answer ?? "",
            payload.source_doc_ids ?? [],
            payload.latency_ms ?? 0,
            payload.refined_query ?? null,
            payload.citations ?? undefined,
          );
        } else if (payload.token !== undefined) {
          callbacks.onToken(payload.token);
        }
      } catch { /* skip malformed */ }
    }
  }
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
 * `GET /docs/{datasetId}/{docId}` → document text by ID.
 * Wired through the gateway to the preprocessing service's
 * `ir_datasets` store for O(1) lookup.
 */
export function fetchDoc(
  datasetId: DatasetId,
  docId: string
): Promise<DocResponse> {
  return api
    .get<DocResponse>(`/docs/${datasetId}/${encodeURIComponent(docId)}`)
    .then((r) => r.data);
}

/**
 * `GET /health` (gateway root, not /api/health). Returns 200 with
 * a `services` map. Useful for the "is the gateway up?" banner.
 */
export function health(): Promise<GatewayHealthResponse> {
  // The gateway exposes /health on its root (not under /api), so we
  // bypass the `/api` baseURL and hit the bare URL.
  return axios
    .get<GatewayHealthResponse>("/health", { timeout: 5_000 })
    .then((r) => r.data);
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
