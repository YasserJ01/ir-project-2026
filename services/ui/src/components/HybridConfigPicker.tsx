/**
 * `HybridConfigPicker` — radio group for the 3 fusion methods
 * supported by the parallel-hybrid path: RRF, CombSUM, CombMNZ.
 *
 * Only visible when the user picks `hybrid_parallel` in the
 * `RepresentationPicker`. The home page wraps the visibility check
 * (so the component itself is dumb).
 *
 * RRF (Reciprocal Rank Fusion, k=60) is the default per
 * Cormack et al. 2009 and the guide §5.3.
 */

import { useUiStore, type FusionMethod } from "../store/useUiStore";

const FUSIONS: { value: FusionMethod; label: string; hint: string }[] = [
  {
    value: "rrf",
    label: "RRF (k=60)",
    hint: "Reciprocal Rank Fusion; default for parallel hybrid.",
  },
  {
    value: "combsum",
    label: "CombSUM",
    hint: "Sum of min-max-normalised scores per retriever.",
  },
  {
    value: "combmnz",
    label: "CombMNZ",
    hint: "CombSUM weighted by the number of retrievers that scored the doc.",
  },
];

export default function HybridConfigPicker() {
  const fusion = useUiStore((s) => s.fusion);
  const setFusion = useUiStore((s) => s.setFusion);

  return (
    <fieldset className="block">
      <legend className="block text-sm font-medium text-slate-700 dark:text-slate-200">
        Fusion (parallel hybrid only)
      </legend>
      <div className="mt-1 grid grid-cols-1 gap-2 sm:grid-cols-3">
        {FUSIONS.map((f) => (
          <label
            key={f.value}
            className={`flex cursor-pointer flex-col rounded-md border p-2 transition ${
              fusion === f.value
                ? "border-indigo-500 bg-indigo-50 ring-1 ring-indigo-500"
                : "border-slate-300 bg-white hover:border-slate-400 dark:border-slate-600 dark:bg-slate-800"
            }`}
          >
            <div className="flex items-center gap-2">
              <input
                type="radio"
                name="fusion"
                value={f.value}
                checked={fusion === f.value}
                onChange={() => setFusion(f.value)}
                className="h-4 w-4 text-indigo-600"
              />
              <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                {f.label}
              </span>
            </div>
            <span className="ml-6 text-xs text-slate-500 dark:text-slate-400">{f.hint}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
