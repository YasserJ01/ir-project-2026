import { useCallback, useState } from "react";
import type { DatasetId, SearchHit } from "../types/api";
import { fetchDoc } from "../api/client";
import { highlight, snippet } from "../utils/highlight";
import { useUserLog } from "../hooks/useUserLog";
import { useUiStore } from "../store/useUiStore";

interface Props {
  hit: SearchHit;
  query: string;
  datasetId: DatasetId;
  highlightTerms: string[];
}

export default function ResultCard({
  hit,
  query,
  datasetId,
  highlightTerms,
}: Props) {
  const userId = useUiStore((s) => s.userId);
  const log = useUserLog();
  const [expanded, setExpanded] = useState(false);
  const [docText, setDocText] = useState<string | null>(null);
  const [loadingDoc, setLoadingDoc] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);

  const preview = snippet(
    `[Document ${hit.doc_id} from ${datasetId}. Score ${hit.score.toFixed(4)}. ` +
      `Click "View" to read the full text.]`,
    280
  );

  const onToggle = useCallback(() => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    // Log the click on first expand.
    log.mutate({
      user_id: userId,
      query,
      doc_id: hit.doc_id,
      dataset_id: datasetId,
      ts: Date.now() / 1000,
    });
    setExpanded(true);
    if (docText !== null || docError !== null) return;
    setLoadingDoc(true);
    setDocError(null);
    fetchDoc(datasetId, hit.doc_id)
      .then((doc) => {
        setDocText(doc.text);
        setLoadingDoc(false);
      })
      .catch((err: Error) => {
        setDocError(err.message);
        setLoadingDoc(false);
      });
  }, [expanded, log, userId, query, hit.doc_id, datasetId, docText, docError]);

  return (
    <li className="rounded-md border border-slate-200 bg-white shadow-sm transition hover:border-indigo-300">
      <div className="flex items-start gap-3 p-3">
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
          onClick={onToggle}
          className="shrink-0 rounded-md border border-indigo-600 px-3 py-1 text-xs font-semibold text-indigo-600 transition hover:bg-indigo-50"
        >
          {expanded ? "Collapse" : "View"}
        </button>
      </div>
      {expanded && (
        <div className="border-t border-slate-200 px-3 pb-3">
          {loadingDoc && (
            <p className="mt-2 animate-pulse text-sm text-slate-500">
              Loading document text…
            </p>
          )}
          {docError && (
            <p className="mt-2 text-sm text-red-600">
              Failed to load document: {docError}
            </p>
          )}
          {docText && (
            <pre className="mt-2 max-h-96 overflow-y-auto whitespace-pre-wrap break-words rounded-md bg-slate-50 p-3 font-mono text-xs leading-relaxed text-slate-800">
              {docText}
            </pre>
          )}
        </div>
      )}
    </li>
  );
}
