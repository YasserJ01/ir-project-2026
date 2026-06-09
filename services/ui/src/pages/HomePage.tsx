/**
 * `HomePage` — the single-page React app. Composes the 9 controls
 * listed in the guide §7.7:
 *
 *   - DatasetSelector
 *   - ModeToggle
 *   - RepresentationPicker
 *   - HybridConfigPicker  (visible only when representation == hybrid_parallel)
 *   - Bm25Sliders         (visible for bm25 / tfidf / hybrid_*)
 *   - SearchBar
 *   - LatencyBadge
 *   - ResultsList
 *   - RagPanel
 *
 * The page owns the local `query` state (so keystrokes don't
 * trigger a re-fetch) and passes it + the zustand `dataset / mode /
 * representation / fusion / bm25 / userId` into `useSearch`, which
 * does the actual fetch and caching.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearch, errorMessage } from "../hooks/useSearch";
import { useDarkMode } from "../hooks/useDarkMode";
import { useUiStore } from "../store/useUiStore";
import type { SearchRequest } from "../types/api";
import DatasetSelector from "../components/DatasetSelector";
import ModeToggle from "../components/ModeToggle";
import RepresentationPicker from "../components/RepresentationPicker";
import HybridConfigPicker from "../components/HybridConfigPicker";
import Bm25Sliders from "../components/Bm25Sliders";
import SearchBar from "../components/SearchBar";
import LatencyBadge from "../components/LatencyBadge";
import ResultsList from "../components/ResultsList";
import RagPanel from "../components/RagPanel";

const SAMPLE_QUERIES = [
  "What is the capital of France?",
  "How does BM25 differ from TF-IDF?",
  "Best programming language for ML",
  "Climate change effects on coral reefs",
];

export default function HomePage() {
  const dataset = useUiStore((s) => s.dataset);
  const mode = useUiStore((s) => s.mode);
  const representation = useUiStore((s) => s.representation);
  const fusion = useUiStore((s) => s.fusion);
  const bm25 = useUiStore((s) => s.bm25);
  const userId = useUiStore((s) => s.userId);
  const [dark, toggleDark] = useDarkMode();

  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const resultsRef = useRef<HTMLDivElement>(null);

  // The `SearchRequest` body. We include bm25 k1/b for the lexical
  // and hybrid paths; the backend ignores them for `embedding`.
  // Phase 7: GatewaySearchRequest accepts bm25_k1 / bm25_b; the
  // gateway forwards them to the indexing service for `tfidf`/`bm25`
  // and to the hybrid endpoint (which has them in
  // HybridSearchRequest) for the hybrid paths.
  const req: SearchRequest = useMemo(
    () => ({
      query: submitted,
      dataset_id: dataset,
      representation,
      k: 10,
      mode,
      fusion,
      user_id: userId,
      enable_grammar: false,
      bm25_k1: bm25.k1,
      bm25_b: bm25.b,
    }),
    [submitted, dataset, representation, mode, fusion, userId, bm25.k1, bm25.b]
  );

  const { data, isFetching, error, refetch } = useSearch(req);

  useEffect(() => {
    if (data && resultsRef.current) {
      resultsRef.current.focus();
    }
  }, [data]);

  const onSubmit = () => {
    setSubmitted(query);
    // refetch immediately in case useSearch hasn't seen the new key yet
    // (it has, but this is a no-op in practice — kept for explicitness).
    void refetch();
  };

  // Highlight terms: prefer the refined query (after spell+synonym
  // expansion) if the backend returned one, else fall back to the
  // raw query. The refined_query is only present when mode=with_features.
  const highlightTerms = useMemo(
    () =>
      (data?.refined_query ?? submitted)
        .split(/\s+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 1),
    [data?.refined_query, submitted]
  );

  const showBm25 = representation !== "embedding";
  const showFusion = representation === "hybrid_parallel";

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <header className="border-b border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <div className="mx-auto flex max-w-5xl items-baseline justify-between p-4">
          <h1 className="text-2xl font-bold">IR Search Engine — 2026</h1>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={toggleDark}
              className="text-xs text-slate-500 hover:underline"
              aria-label="Toggle dark mode"
            >
              {dark ? "☀️ light" : "🌙 dark"}
            </button>
            <a
              href="/health"
              className="text-xs text-slate-500 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              gateway health
            </a>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-5xl space-y-4 p-4 sm:p-6">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <DatasetSelector />
          <ModeToggle />
        </div>
        <RepresentationPicker />
        {showFusion && <HybridConfigPicker />}
        {showBm25 && <Bm25Sliders />}
        <SearchBar
          value={query}
          onChange={setQuery}
          onSubmit={onSubmit}
          loading={isFetching}
        />
        <div className="flex items-center justify-between">
          <LatencyBadge
            ms={data?.latency_ms}
            fellBack={data?.refinement_fell_back}
          />
          <span className="text-xs text-slate-500" aria-live="polite">
            {data ? `${data.results.length} results` : ""}
          </span>
        </div>

        {submitted.length === 0 ? (
          <div className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-700 dark:bg-slate-800">
            <p className="mb-2 text-slate-600">
              Try one of these sample queries:
            </p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => {
                    setQuery(q);
                    setSubmitted(q);
                  }}
                  className="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs text-slate-700 transition hover:border-indigo-400 hover:text-indigo-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div ref={resultsRef} tabIndex={-1}>
            <ResultsList
              hits={data?.results ?? []}
              loading={isFetching}
              error={error ? errorMessage(error) : null}
              query={submitted}
              datasetId={dataset}
              highlightTerms={highlightTerms}
            />
          </div>
        )}

        <RagPanel query={submitted} dataset={dataset} enabled={submitted.length > 0} />
      </section>

      <footer className="mx-auto max-w-5xl p-4 text-center text-xs text-slate-400 dark:text-slate-500">
        Phase 7 · React 18 + Vite 5 + TypeScript 5 + Tailwind 3 ·
        {" "}
        <a
          className="hover:underline"
          href="https://github.com/YasserJ01/ir-project-2026"
          target="_blank"
          rel="noopener noreferrer"
        >
          github.com/YasserJ01/ir-project-2026
        </a>
      </footer>
    </main>
  );
}
