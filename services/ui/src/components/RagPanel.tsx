/**
 * `RagPanel` — RAG answer panel (Phase 8).
 *
 * Behaviour:
 *   - The "Get an answer" button is enabled iff there's a non-empty
 *     query and a dataset.
 *   - Clicking it calls `ragAnswer(...)` and renders the answer +
 *     source doc_ids.
 *   - Errors are surfaced as a red banner.
 */

import { useEffect, useRef, useState } from "react";
import { ragAnswer, ragAnswerStream, errorMessage } from "../api/client";
import type { DatasetId } from "../types/api";

interface Props {
  query: string;
  dataset: DatasetId;
  /** When false, the panel is collapsed and the button is hidden. */
  enabled: boolean;
}

const RETRIEVERS = [
  { value: "embedding" as const, label: "Embedding", hint: "Best semantic match" },
  { value: "hybrid_parallel" as const, label: "Hybrid", hint: "Lexical + semantic" },
  { value: "bm25" as const, label: "BM25", hint: "Fast, lexical only" },
];

export default function RagPanel({ query, dataset, enabled }: Props) {
  const [open, setOpen] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [retriever, setRetriever] = useState<"bm25" | "embedding" | "hybrid_parallel">("embedding");
  const [refinedQuery, setRefinedQuery] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(true);
  const [stage, setStage] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setOpen(false);
  }, [query]);

  if (!enabled) return null;

  const canAsk = query.trim().length > 0 && !loading;

  async function onAsk() {
    setOpen(true);
    setLoading(true);
    setErr(null);
    setAnswer(null);
    setSources([]);
    setRefinedQuery(null);
    setStage(null);

    if (!streaming) {
      try {
        const r = await ragAnswer({ query, dataset_id: dataset, k: 5, max_tokens: 256, retriever });
        setAnswer(r.answer ?? "(no answer returned)");
        setSources(r.source_doc_ids ?? []);
        setRefinedQuery(r.refined_query ?? null);
      } catch (e) {
        setErr(errorMessage(e));
      } finally {
        setLoading(false);
      }
      return;
    }

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    await ragAnswerStream(
      { query, dataset_id: dataset, k: 5, max_tokens: 256, retriever },
      {
        onStage: (s, data) => {
          setStage(s);
          if (data.source_doc_ids) setSources(data.source_doc_ids as string[]);
          if (data.refined_query) setRefinedQuery(data.refined_query as string);
        },
        onToken: (token) => {
          setAnswer((prev) => (prev ?? "") + token);
        },
        onDone: (ans, srcs, _latency, refined) => {
          setAnswer(ans);
          setSources(srcs);
          setRefinedQuery(refined);
          setLoading(false);
          setStage(null);
        },
        onError: (msg) => {
          setErr(msg);
          setLoading(false);
          setStage(null);
        },
      },
      ctrl.signal,
    );
  }

  return (
    <section className="rounded-md border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-600 dark:bg-slate-800">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          RAG answer
        </h2>
        <div className="flex items-center gap-2">
          <select
            value={retriever}
            onChange={(e) => setRetriever(e.target.value as "bm25" | "embedding" | "hybrid_parallel")}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            aria-label="Retrieval method"
          >
            {RETRIEVERS.map((r) => (
              <option key={r.value} value={r.value} title={r.hint}>
                {r.label}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
              className="h-3 w-3 rounded border-slate-300 text-indigo-600"
            />
            Stream
          </label>
          <button
            type="button"
            disabled={!canAsk}
            onClick={onAsk}
            className="rounded-md bg-emerald-600 px-3 py-1 text-xs font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Asking…" : "Get an answer"}
          </button>
        </div>
      </div>
      {open && (
        <div className="mt-2 space-y-2 text-sm">
          {refinedQuery && refinedQuery !== query && (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Expanded from: <span className="italic">{query}</span> →{" "}
              <span className="font-medium">{refinedQuery}</span>
            </p>
          )}
          {stage && (
            <p className="text-xs text-slate-400 dark:text-slate-500">
              {stage === "retrieval" ? "Retrieving documents…" : stage}
            </p>
          )}
          {loading && !stage && (
            <p className="animate-pulse text-slate-500 dark:text-slate-400">
              {streaming ? "Generating…" : "Generating answer…"}
            </p>
          )}
          {err && (
            <p
              role="alert"
              className="rounded-md border border-amber-200 bg-amber-50 p-2 text-amber-900"
            >
              {err}
            </p>
          )}
          {answer && (
            <div>
              <p className="text-slate-800 dark:text-slate-100">{answer}</p>
              {sources.length > 0 && (
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  Sources: {sources.map((s) => s).join(", ")}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
