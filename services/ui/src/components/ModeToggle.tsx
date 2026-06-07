/**
 * `ModeToggle` — radio group for `basic` vs `with_features`.
 * `with_features` triggers an upstream call to the refinement
 * service for spell + synonyms + grammar + personalization.
 *
 * Bound to zustand `mode` field; the home page passes `mode` into
 * the `SearchRequest` body so the gateway routes accordingly.
 */

import { useUiStore } from "../store/useUiStore";
import type { SearchMode } from "../types/api";

const MODES: { value: SearchMode; label: string; hint: string }[] = [
  {
    value: "basic",
    label: "Basic",
    hint: "Send the raw query directly to the retriever.",
  },
  {
    value: "with_features",
    label: "With features",
    hint:
      "Spell-correct, expand synonyms, apply grammar (optional), and personalize weights from your click history.",
  },
];

export default function ModeToggle() {
  const mode = useUiStore((s) => s.mode);
  const setMode = useUiStore((s) => s.setMode);

  return (
    <fieldset className="block">
      <legend className="block text-sm font-medium text-slate-700">
        Mode
      </legend>
      <div className="mt-1 grid grid-cols-1 gap-2 sm:grid-cols-2">
        {MODES.map((m) => (
          <label
            key={m.value}
            className={`flex cursor-pointer flex-col rounded-md border p-2 transition ${
              mode === m.value
                ? "border-indigo-500 bg-indigo-50 ring-1 ring-indigo-500"
                : "border-slate-300 bg-white hover:border-slate-400"
            }`}
          >
            <div className="flex items-center gap-2">
              <input
                type="radio"
                name="mode"
                value={m.value}
                checked={mode === m.value}
                onChange={() => setMode(m.value)}
                className="h-4 w-4 text-indigo-600"
              />
              <span className="text-sm font-semibold text-slate-800">
                {m.label}
              </span>
            </div>
            <span className="ml-6 text-xs text-slate-500">{m.hint}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
