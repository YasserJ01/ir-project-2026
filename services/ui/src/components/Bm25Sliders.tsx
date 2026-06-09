/**
 * `Bm25Sliders` — two range inputs (k1, b) bound to the zustand
 * store. Changes are debounced 300 ms before being committed, so
 * dragging the thumb doesn't re-fetch on every tick.
 *
 * Hidden by the home page when the chosen `representation` is not
 * BM25-based (i.e. only `embedding`); visible for the lexical
 * (`tfidf` still allows tweaking) and hybrid paths.
 *
 * - k1 (BM25 term-frequency saturation): typical 1.2-2.0; default 1.5
 * - b  (BM25 length normalization):   0.0-1.0;      default 0.75
 */

import { useEffect, useRef, useState } from "react";
import { useUiStore } from "../store/useUiStore";

const DEBOUNCE_MS = 300;

export default function Bm25Sliders() {
  const bm25 = useUiStore((s) => s.bm25);
  const setBm25 = useUiStore((s) => s.setBm25);
  const reset = useUiStore((s) => s.resetBm25);

  // Local state for the input value, committed to the store
  // after a debounce. The sliders stay responsive; the consumer
  // (useSearch via HomePage) re-fetches once.
  const [local, setLocal] = useState(bm25);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setLocal(bm25);
  }, [bm25]);

  function commit(next: typeof bm25, immediate = false) {
    setLocal(next);
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
    if (immediate) {
      setBm25(next);
    } else {
      timer.current = window.setTimeout(() => {
        setBm25(next);
      }, DEBOUNCE_MS);
    }
  }

  return (
    <div className="block rounded-md border border-slate-200 bg-white p-3 dark:border-slate-600 dark:bg-slate-800">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700 dark:text-slate-200">BM25</span>
        <button
          type="button"
          onClick={() => {
            const defaults = { k1: 1.5, b: 0.75 };
            commit(defaults, true);
            reset();
          }}
          className="text-xs font-medium text-indigo-600 hover:underline"
        >
          Reset
        </button>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="block">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-600 dark:text-slate-300">k1 (term-frequency saturation)</span>
            <span className="font-mono text-xs text-slate-700 dark:text-slate-100">
              {local.k1.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={3}
            step={0.05}
            value={local.k1}
            onChange={(e) =>
              commit({ ...local, k1: Number(e.target.value) })
            }
            className="mt-1 w-full accent-indigo-600"
          />
        </label>
        <label className="block">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-600 dark:text-slate-300">b (length normalization)</span>
            <span className="font-mono text-xs text-slate-700 dark:text-slate-100">
              {local.b.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={local.b}
            onChange={(e) =>
              commit({ ...local, b: Number(e.target.value) })
            }
            className="mt-1 w-full accent-indigo-600"
          />
        </label>
      </div>
    </div>
  );
}
