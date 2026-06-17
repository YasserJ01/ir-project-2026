/**
 * `EvalPanel` — run a full evaluation and display metrics.
 *
 * Shows a "Run Evaluation" button that loads all sampled queries,
 * runs them through the current configuration, and displays the
 * results table with colored indicators. Supports Excel export.
 */

import { useCallback, useState } from "react";
import { runEvaluation } from "../api/client";
import type { DatasetId, EvaluateResponse, Representation, SearchMode, FusionMethod } from "../types/api";

interface Props {
  dataset: DatasetId;
  representation: Representation;
  mode: SearchMode;
  fusion: FusionMethod;
  bm25K1: number;
  bm25B: number;
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function colorClass(value: number): string {
  if (value >= 0.4) return "text-green-600 dark:text-green-400 font-semibold";
  if (value >= 0.2) return "text-amber-600 dark:text-amber-400";
  return "text-red-500 dark:text-red-400";
}

function exportToExcel(data: EvaluateResponse): void {
  const header = "Metric,Value\n";
  const rows = [
    `Dataset,${data.dataset_id}\n`,
    `Representation,${data.representation}\n`,
    `Condition,${data.condition}\n`,
    `Queries,${data.queries}\n`,
    `Success,${data.success}\n`,
    `Errors,${data.errors}\n`,
    `Time (s),${data.time_s}\n`,
    `MAP,${data.metrics.MAP}\n`,
    `P@10,${data.metrics["P@10"]}\n`,
    `nDCG@10,${data.metrics["nDCG@10"]}\n`,
    `R@10,${data.metrics["R@10"]}\n`,
  ];
  const csv = header + rows.join("");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `eval_${data.dataset_id}_${data.representation}_${data.condition}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function EvalPanel({ dataset, representation, mode, fusion, bm25K1, bm25B }: Props) {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setProgress("Loading queries and warming caches...");
    setResult(null);
    setError(null);

    try {
      const resp = await runEvaluation({
        dataset_id: dataset,
        representation,
        mode,
        fusion,
        bm25_k1: bm25K1,
        bm25_b: bm25B,
        use_multi: false,
      });
      if ("error" in resp && typeof resp.error === "string") {
        setError(resp.error as string);
      } else {
        setResult(resp as EvaluateResponse);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Evaluation failed";
      setError(msg);
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }, [dataset, representation, mode, fusion, bm25K1, bm25B]);

  const handleExport = useCallback(() => {
    if (result) exportToExcel(result);
  }, [result]);

  return (
    <div className="rounded-md border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-300">
        Live Evaluation
      </h3>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={handleRun}
          disabled={loading}
          className="rounded bg-indigo-600 px-3 py-1 text-xs text-white transition hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? "Running..." : "Run Evaluation"}
        </button>
        {result && (
          <button
            type="button"
            onClick={handleExport}
            className="rounded bg-emerald-600 px-3 py-1 text-xs text-white transition hover:bg-emerald-700"
          >
            Export CSV
          </button>
        )}
      </div>

      {progress && (
        <p className="mt-2 text-xs text-slate-400">{progress}</p>
      )}

      {error && (
        <p className="mt-2 text-xs text-red-500">{error}</p>
      )}

      {result && (
        <div className="mt-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-700">
                <th className="py-1 pr-3 text-left font-medium text-slate-500">Metric</th>
                <th className="py-1 text-right font-medium text-slate-500">Value</th>
              </tr>
            </thead>
            <tbody>
              {(["MAP", "P@10", "nDCG@10", "R@10"] as const).map((m) => (
                <tr key={m} className="border-b border-slate-100 dark:border-slate-700/50">
                  <td className="py-1 pr-3 text-slate-600 dark:text-slate-400">{m}</td>
                  <td className={`py-1 text-right font-mono ${colorClass(result.metrics[m])}`}>
                    {pct(result.metrics[m])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mt-2 flex gap-4 text-[10px] text-slate-400">
            <span>{result.queries} queries</span>
            <span>{result.success} ok</span>
            {result.errors > 0 && <span className="text-red-400">{result.errors} err</span>}
            <span>{result.time_s}s</span>
          </div>

          <div className="mt-2 flex gap-4 rounded bg-slate-50 p-2 text-[10px] dark:bg-slate-700/50">
            <span className="text-slate-500">
              Configuration: {result.dataset_id} / {result.representation} / {result.condition}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
