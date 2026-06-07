/**
 * `LatencyBadge` — small "⏱ 312 ms" pill. Shown next to the search
 * bar so the user can see how long the last request took. Returns
 * nothing when `ms` is undefined.
 */

interface Props {
  ms: number | undefined;
  fellBack?: boolean;
}

export default function LatencyBadge({ ms, fellBack }: Props) {
  if (ms === undefined) {
    return (
      <div className="text-xs text-slate-400" aria-live="polite">
        —
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500" aria-live="polite">
      <span
        className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 font-mono text-slate-700"
        title="End-to-end latency reported by the gateway"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className="h-3 w-3"
          aria-hidden
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .2.08.39.22.53l3 3a.75.75 0 101.06-1.06L10.75 9.69V5z"
            clipRule="evenodd"
          />
        </svg>
        <span>{ms} ms</span>
      </span>
      {fellBack && (
        <span
          className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-800"
          title="Refinement service unreachable; ran in basic mode"
        >
          fell back to basic
        </span>
      )}
    </div>
  );
}
