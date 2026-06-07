/**
 * `RagPanel` — RAG answer panel. For Phase 7 the gateway returns
 * 501 (RAG ships in Phase 8); the panel UI is in place so the home
 * page can render the "Get an answer" button + answer + sources
 * state once Phase 8 lands.
 *
 * Behaviour:
 *   - The "Get an answer" button is enabled iff there's a non-empty
 *     query and a dataset.
 *   - Clicking it calls `ragAnswer(...)` and renders the answer +
 *     source doc_ids.
 *   - Errors are surfaced as a red banner.
 */

import { useState } from "react";
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
    <section className="rounded-md border border-slate-200 bg-white p-3 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-800">
          RAG answer (Phase 8 preview)
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
            <p className="animate-pulse text-slate-500">
              Calling RAG service (currently 501 stub)…
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
              <p className="text-slate-800">{answer}</p>
              {sources.length > 0 && (
                <p className="mt-2 text-xs text-slate-500">
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
