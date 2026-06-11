/**
 * Global UI state (Zustand). Persisted to localStorage so a page
 * reload keeps the user's dataset / mode / BM25 sliders.
 *
 * Mirrors the `Representation` / `FusionMethod` / `SearchMode`
 * Literal types from `shared/ir_common/schemas.py`. The TS types
 * themselves live in `types/api.ts`.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DatasetId, FusionMethod, Representation, SearchMode } from "../types/api";

export type { DatasetId, FusionMethod, Representation, SearchMode };

interface Bm25 {
  k1: number;
  b: number;
}

interface UiState {
  /** Active corpus. Persisted. */
  dataset: DatasetId;
  /** "basic" skips the refinement service; "with_features" enables
   *  spell + synonyms + grammar + personalization. */
  mode: SearchMode;
  /** Which retriever(s) to run. */
  representation: Representation;
  /** Fusion method (parallel hybrid only). */
  fusion: FusionMethod;
  /** BM25 hyper-parameters. Debounced 300 ms before re-fetching. */
  bm25: Bm25;
  /** Per-browser persistent user id. Random UUID on first visit. */
  userId: string;
  /** Cluster boost on/off. */
  enableClustering: boolean;
  /** Cluster boost multiplier (1.0–3.0). */
  clusterBoost: number;
  setDataset: (d: DatasetId) => void;
  setMode: (m: SearchMode) => void;
  setRepresentation: (r: Representation) => void;
  setFusion: (f: FusionMethod) => void;
  setBm25: (b: Bm25) => void;
  resetBm25: () => void;
  setEnableClustering: (v: boolean) => void;
  setClusterBoost: (v: number) => void;
}

const DEFAULT_BM25: Bm25 = { k1: 1.5, b: 0.75 };

/** Crypto-strong UUID v4 generator with a fallback for old browsers. */
function newUserId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback: time + random bytes; uniqueness is good enough for per-user logs.
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      dataset: "touche2020",
      mode: "basic",
      representation: "bm25",
      fusion: "rrf",
      bm25: { ...DEFAULT_BM25 },
      userId: newUserId(),
      enableClustering: false,
      clusterBoost: 1.5,
      setDataset: (dataset) => set({ dataset }),
      setMode: (mode) => set({ mode }),
      setRepresentation: (representation) => set({ representation }),
      setFusion: (fusion) => set({ fusion }),
      setBm25: (bm25) => set({ bm25 }),
      resetBm25: () => set({ bm25: { ...DEFAULT_BM25 } }),
      setEnableClustering: (v) => set({ enableClustering: v }),
      setClusterBoost: (v) => set({ clusterBoost: v }),
    }),
    {
      name: "ir-ui",
      // userId is a stable per-browser identifier; it must NOT be
      // regenerated on every page load, but it must also be created
      // once. The persist middleware will write the initial UUID to
      // localStorage on the first store update.
      partialize: (s) => ({
        dataset: s.dataset,
        mode: s.mode,
        representation: s.representation,
        fusion: s.fusion,
        bm25: s.bm25,
        userId: s.userId,
      }),
    }
  )
);
