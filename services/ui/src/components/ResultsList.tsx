/**
 * `ResultsList` — the list of `ResultCard`s. Shown after a search
 * completes. Empty / loading / error states are all handled here so
 * the home page stays a thin assembler.
 *
 * Loading: a single full-width skeleton card.
 * Error: red banner with the error message.
 * Empty: "No results — try a different query."
 */

import type { SearchHit } from "../types/api";
import ResultCard from "./ResultCard";

interface Props {
  hits: SearchHit[];
  loading: boolean;
  error: string | null;
  query: string;
  datasetId: string;
  highlightTerms: string[];
}

export default function ResultsList({
  hits,
  loading,
  error,
  query,
  datasetId,
  highlightTerms,
}: Props) {
  if (loading) {
    return (
      <ul className="space-y-2" aria-busy="true">
        {[0, 1, 2].map((i) => (
          <li
            key={i}
            className="h-20 animate-pulse rounded-md border border-slate-200 bg-slate-100"
          />
        ))}
      </ul>
    );
  }

  if (error) {
    return (
      <div
        role="alert"
        className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-800"
      >
        <strong className="font-semibold">Search failed.</strong> {error}
      </div>
    );
  }

  if (hits.length === 0) {
    return (
      <div className="rounded-md border border-slate-200 bg-white p-3 text-sm text-slate-500">
        No results. Try a different query or representation.
      </div>
    );
  }

  return (
    <ul className="space-y-2" aria-label="Search results">
      {hits.map((hit) => (
        <ResultCard
          key={hit.doc_id}
          hit={hit}
          query={query}
          datasetId={datasetId}
          highlightTerms={highlightTerms}
        />
      ))}
    </ul>
  );
}
