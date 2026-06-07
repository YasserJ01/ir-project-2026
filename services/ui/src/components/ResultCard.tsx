/**
 * `ResultCard` — one row in the results list. Shows rank, doc_id,
 * a snippet (placeholder text — we don't have a per-doc preview
 * endpoint yet; the snippet is the query context as a fallback),
 * the score, and a "View" button that fires the click log and
 * opens the original doc in a new tab (anchor target).
 *
 * Click logging is fire-and-forget: see `useUserLog`. We don't
 * `await` it because the user shouldn't wait for the log to
 * complete before navigating.
 */

import type { SearchHit } from "../types/api";
import { highlight, snippet } from "../utils/highlight";
import { useUserLog } from "../hooks/useUserLog";
import { useUiStore } from "../store/useUiStore";

interface Props {
  hit: SearchHit;
  query: string;
  datasetId: string;
  highlightTerms: string[];
  // Optional: a real preview snippet per doc. We don't have a
  // /docs/{id} endpoint yet; if the caller has one, pass it.
  docSnippet?: string;
  // Optional: a real URL to open. Defaults to a search-engine query
  // for the doc_id (so the "View" link doesn't 404 during Phase 7).
  docUrl?: (docId: string) => string;
}

export default function ResultCard({
  hit,
  query,
  datasetId,
  highlightTerms,
  docSnippet,
  docUrl,
}: Props) {
  const userId = useUiStore((s) => s.userId);
  const log = useUserLog();
  const preview =
    docSnippet ??
    snippet(
      `[Document ${hit.doc_id} from ${datasetId}. Score ${hit.score.toFixed(4)}. ` +
        `A per-document preview is not available yet; the gateway returns doc_id + score only. ` +
        `Click "View" to open this document in a new tab.]`,
      280
    );

  function onView() {
    log.mutate({
      user_id: userId,
      query,
      doc_id: hit.doc_id,
      dataset_id: datasetId as typeof hit.doc_id extends string
        ? Parameters<typeof log.mutate>[0]["dataset_id"]
        : never,
      ts: Date.now() / 1000,
    });
    const url =
      docUrl?.(hit.doc_id) ??
      `https://www.google.com/search?q=%22${encodeURIComponent(hit.doc_id)}%22`;
    window.open(url, "_blank", "noopener,noreferrer");
  }

  return (
    <li className="rounded-md border border-slate-200 bg-white p-3 shadow-sm transition hover:border-indigo-300">
      <div className="flex items-start gap-3">
        <span
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-200 text-xs font-semibold text-slate-700"
          aria-label={`Rank ${hit.rank}`}
        >
          {hit.rank}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <code className="break-all font-mono text-xs text-slate-600">
              {hit.doc_id}
            </code>
            <span className="font-mono text-xs text-slate-500">
              score={hit.score.toFixed(4)}
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-700">
            {highlight(preview, highlightTerms)}
          </p>
          {hit.individual_scores &&
            Object.keys(hit.individual_scores).length > 0 && (
              <p className="mt-1 text-xs text-slate-500">
                per-retriever:{" "}
                {Object.entries(hit.individual_scores)
                  .map(([k, v]) => `${k}=${v.toFixed(3)}`)
                  .join(", ")}
              </p>
            )}
        </div>
        <button
          type="button"
          onClick={onView}
          className="shrink-0 rounded-md border border-indigo-600 px-3 py-1 text-xs font-semibold text-indigo-600 transition hover:bg-indigo-50"
        >
          View
        </button>
      </div>
    </li>
  );
}
