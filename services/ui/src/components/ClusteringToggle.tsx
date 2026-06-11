interface Props {
  enabled: boolean;
  boost: number;
  onToggle: (v: boolean) => void;
  onBoostChange: (v: number) => void;
}

export default function ClusteringToggle({ enabled, boost, onToggle, onBoostChange }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border border-slate-200 bg-white p-2 shadow-sm dark:border-slate-600 dark:bg-slate-800">
      <label className="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-300">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-slate-300 text-indigo-600"
        />
        Clustering
      </label>
      {enabled && (
        <label className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
          <span>Boost:</span>
          <input
            type="range"
            min={1.0}
            max={3.0}
            step={0.1}
            value={boost}
            onChange={(e) => onBoostChange(parseFloat(e.target.value))}
            className="h-1.5 w-20 accent-indigo-600"
          />
          <span className="w-6 text-right font-mono text-slate-600 dark:text-slate-300">
            {boost.toFixed(1)}x
          </span>
        </label>
      )}
    </div>
  );
}
