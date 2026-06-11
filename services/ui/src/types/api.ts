/**
 * TypeScript interfaces mirroring `shared/ir_common/schemas.py`
 * (Pydantic v2 models). The React UI sends/receives these via the
 * gateway (port 8000) under `/api/*`.
 *
 * Whenever a Pydantic schema changes in `shared/ir_common/schemas.py`,
 * the corresponding TS interface here must be updated to keep the
 * wire contract in sync. The fields use snake_case to match the
 * Python/JSON wire format directly (no camelCase transform — the
 * gateway and backend services speak snake_case).
 */

export type DatasetId = "touche2020" | "nq";

export type Representation =
  | "tfidf"
  | "bm25"
  | "embedding"
  | "hybrid_serial"
  | "hybrid_parallel";

export type FusionMethod = "rrf" | "combsum" | "combmnz";

export type SearchMode = "basic" | "with_features";

export type AddedBy = "original" | "spell" | "synonym" | "grammar" | "personalization";

/** Body for `POST /api/search` (gateway). */
export interface SearchRequest {
  query: string;
  dataset_id: DatasetId;
  representation?: Representation;
  k?: number;
  mode?: SearchMode;
  fusion?: FusionMethod;
  user_id?: string | null;
  enable_grammar?: boolean;
  /** Phase 7: BM25 k1 slider. Forwarded to the indexing service for
   *  `tfidf`/`bm25` representations and to the hybrid endpoint for
   *  hybrid paths. Ignored for `embedding`. */
  bm25_k1?: number;
  /** Phase 7: BM25 b slider. Same forwarding rules as `bm25_k1`. */
  bm25_b?: number;
}

/** One hit in a `SearchResponse` (BM25 / TF-IDF path) or a hybrid response. */
export interface SearchHit {
  rank: number;
  doc_id: string;
  score: number;
  individual_scores?: Record<string, number>;
}

/** Response from `POST /api/search`. Mirrors `HybridSearchResponse`. */
export interface SearchResponse {
  dataset_id: DatasetId;
  representation: Representation;
  fusion?: FusionMethod | null;
  k: number;
  latency_ms: number;
  results: SearchHit[];
  bm25_k1?: number | null;
  bm25_b?: number | null;
  per_retriever_latency_ms?: Record<string, number>;
  refined_query?: string | null;
  stages?: Record<string, string>;
  refinement_fell_back?: boolean;
}

/** Body for `POST /api/refine`. */
export interface RefineRequest {
  query: string;
  user_id?: string;
  enable_spell?: boolean;
  enable_synonyms?: boolean;
  enable_grammar?: boolean;
  enable_personalization?: boolean;
  synonym_count?: number;
}

/** One token in `RefineResponse.weighted_tokens`. */
export interface RefinedToken {
  token: string;
  weight: number;
  added_by: AddedBy;
}

/** Response from `POST /api/refine`. */
export interface RefineResponse {
  query: string;
  refined_query: string;
  expanded_query: string;
  tokens: string[];
  weighted_tokens: RefinedToken[];
  stages: Record<string, string>;
  latency_ms: number;
  user_id: string;
}

/** Response from `GET /api/datasets`. */
export interface DatasetsResponse {
  datasets: DatasetId[];
}

/** Response from `GET /health` (gateway). */
export interface GatewayHealthResponse {
  status: "ok" | "degraded";
  service: string;
  version: string;
  services: Record<string, boolean>;
}

/** Body for `POST /api/log/click`. */
export interface LogClickRequest {
  user_id: string;
  query: string;
  doc_id: string;
  dataset_id: DatasetId;
  ts?: number | null;
}

/** Body for `POST /api/rag/answer`. */
export interface RagRequest {
  dataset_id: DatasetId;
  query: string;
  k?: number;
  max_tokens?: number;
  retriever?: "bm25" | "embedding" | "hybrid_parallel";
  conversation_id?: string | null;
}

export interface RagResponse {
  answer?: string;
  source_doc_ids?: string[];
  latency_ms?: number;
  refined_query?: string | null;
  /** Map of citation number -> doc_id, e.g. {"1": "doc-abc", "2": "doc-xyz"} */
  citations?: Record<string, string>;
}

/** Response from `GET /api/docs/{dataset_id}/{doc_id}`. */
export interface DocResponse {
  id: string;
  text: string;
}

/** Error body returned by the gateway on 502/503 (mirrors `GatewayErrorResponse`). */
export interface GatewayErrorBody {
  service: string;
  reachable: boolean;
  status_code: number | null;
  detail: string;
}

/** Body for `POST /api/cluster/{ds}/search`. */
export interface ClusterSearchRequest {
  query: string;
  dataset_id: DatasetId;
  representation?: Representation;
  k?: number;
  mode?: SearchMode;
  fusion?: FusionMethod;
  user_id?: string | null;
  enable_grammar?: boolean;
  bm25_k1?: number;
  bm25_b?: number;
  enable_clustering?: boolean;
  cluster_boost?: number;
}

/** Response from `POST /api/cluster/{ds}/search`. */
export interface ClusterSearchResponse {
  results: SearchHit[];
  query: string;
  dataset_id: DatasetId;
  latency_ms: number;
  nearest_cluster_id: number;
  cluster_centroid_distance: number;
  cluster_sizes: number[];
  representation: Representation;
}

/** Axios error augmentation. */
export interface ApiError {
  status: number;
  message: string;
  body?: GatewayErrorBody | { detail: string | GatewayErrorBody } | string;
}
