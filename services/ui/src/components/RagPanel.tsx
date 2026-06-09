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

import { useEffect, useState } from "react";
import { ragAnswer, errorMessage } from "../api/client";
import type { DatasetId } from "../types/api";

interface Props {
  query: string;
  dataset: DatasetId;
  /** When false, the panel is collapsed and the button is hidden. */
  enabled: boolean;
}

export default function RagPanel({ query, dataset, enabled }: Props) {
  const [open, setOpen] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

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
    try {
      const r = await ragAnswer({ query, dataset_id: dataset, k: 5 });
      setAnswer(r.answer ?? "(no answer returned)");
      setSources(r.source_doc_ids ?? []);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rounded-md border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-600 dark:bg-slate-800">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          RAG answer
        </h2>
        <button
          type="button"
          disabled={!canAsk}
          onClick={onAsk}
          className="rounded-md bg-emerald-600 px-3 py-1 text-xs font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {loading ? "Asking…" : "Get an answer"}
        </button>
      </div>
      {open && (
        <div className="mt-2 space-y-2 text-sm">
          {loading && (
            <p className="animate-pulse text-slate-500 dark:text-slate-400">
              Generating answer…
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
