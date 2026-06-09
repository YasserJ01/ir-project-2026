/**
 * `RepresentationPicker` — radio group for the 5 representations
 * supported by the gateway's `POST /api/search`:
 *   tfidf, bm25, embedding, hybrid_serial, hybrid_parallel.
 *
 * Changing the representation updates the zustand store; the home
 * page reads it and passes it into the `SearchRequest`.
 *
 * The `hybrid_parallel` option shows the "Fusion" picker (handled
 * by a separate `HybridConfigPicker` component on the home page).
 */

import { useUiStore, type Representation } from "../store/useUiStore";

const REPS: { value: Representation; label: string; hint: string }[] = [
  {
    value: "tfidf",
    label: "TF-IDF",
    hint: "Lexical baseline (sklearn TfidfVectorizer).",
  },
  {
    value: "bm25",
    label: "BM25",
    hint: "Lexical best-practice (bm25s, Lucene variant).",
  },
  {
    value: "embedding",
    label: "Embedding",
    hint: "Single dense encoder (MiniLM-L6).",
  },
  {
    value: "hybrid_serial",
    label: "Hybrid serial",
    hint: "BM25 top-1000 → dense re-rank top-k.",
  },
  {
    value: "hybrid_parallel",
    label: "Hybrid parallel",
    hint: "Run {BM25, dense} in parallel and fuse.",
  },
];

export default function RepresentationPicker() {
  const representation = useUiStore((s) => s.representation);
  const setRepresentation = useUiStore((s) => s.setRepresentation);

  return (
    <fieldset className="block">
      <legend className="block text-sm font-medium text-slate-700 dark:text-slate-200">
        Representation
      </legend>
      <div className="mt-1 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {REPS.map((r) => (
          <label
            key={r.value}
            className={`flex cursor-pointer flex-col rounded-md border p-2 transition ${
              representation === r.value
                ? "border-indigo-500 bg-indigo-50 ring-1 ring-indigo-500"
                : "border-slate-300 bg-white hover:border-slate-400 dark:border-slate-600 dark:bg-slate-800"
            }`}
          >
            <div className="flex items-center gap-2">
              <input
                type="radio"
                name="representation"
                value={r.value}
                checked={representation === r.value}
                onChange={() => setRepresentation(r.value)}
                className="h-4 w-4 text-indigo-600"
              />
              <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                {r.label}
              </span>
            </div>
            <span className="ml-6 text-xs text-slate-500 dark:text-slate-400">{r.hint}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
