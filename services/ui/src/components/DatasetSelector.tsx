/**
 * `DatasetSelector` — dropdown bound to the canonical list returned
 * by `GET /api/datasets`. The chosen value lives in the zustand
 * store (persisted to localStorage as `dataset`).
 *
 * The dropdown is disabled while the list is loading, and shows a
 * "—" placeholder if the gateway is unreachable (useDatasets will
 * return an error after the 30 s axios timeout).
 */

import { useDatasets } from "../hooks/useDatasets";
import { useUiStore } from "../store/useUiStore";

export default function DatasetSelector() {
  const dataset = useUiStore((s) => s.dataset);
  const setDataset = useUiStore((s) => s.setDataset);
  const { data: datasets, isLoading, isError, error } = useDatasets();

  return (
    <label className="block">
      <span className="block text-sm font-medium text-slate-700 dark:text-slate-200">
        Dataset
      </span>
      <select
        className="mt-1 w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:bg-slate-100 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        value={dataset}
        onChange={(e) => setDataset(e.target.value as typeof dataset)}
        disabled={isLoading || isError}
      >
        {isLoading && <option value="">Loading datasets…</option>}
        {isError && (
          <option value="">
            Gateway unreachable
            {error ? ` (${error.message})` : ""}
          </option>
        )}
        {datasets?.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>
      <span className="mt-1 block text-xs text-slate-500 dark:text-slate-400">
        Choose the corpus to search.
      </span>
    </label>
  );
}
