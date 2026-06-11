interface Props {
  sizes: number[];
  nearest?: number;
}

const BAR_COLORS = [
  "bg-indigo-500",
  "bg-emerald-500",
  "bg-amber-500",
  "bg-rose-500",
  "bg-cyan-500",
  "bg-violet-500",
  "bg-lime-500",
  "bg-pink-500",
  "bg-teal-500",
  "bg-orange-500",
  "bg-blue-500",
  "bg-green-500",
  "bg-yellow-500",
  "bg-red-500",
  "bg-purple-500",
  "bg-sky-500",
  "bg-fuchsia-500",
  "bg-stone-500",
  "bg-slate-500",
  "bg-gray-500",
];

export default function ClusterBarChart({ sizes, nearest }: Props) {
  if (sizes.length === 0) return null;
  const maxVal = Math.max(...sizes, 1);

  return (
    <div className="text-xs">
      <p className="mb-1 font-semibold text-slate-600 dark:text-slate-300">
        Cluster sizes
      </p>
      <div className="flex h-16 items-end gap-0.5">
        {sizes.map((s, i) => {
          const hPct = (s / maxVal) * 100;
          const isNearest = i === nearest;
          return (
            <div
              key={i}
              className="group relative flex flex-1 flex-col items-center"
              title={`Cluster ${i}: ${s.toLocaleString()} docs${isNearest ? " (nearest)" : ""}`}
            >
              <div
                className={`w-full rounded-t ${
                  isNearest ? "bg-amber-400 ring-1 ring-amber-600" : BAR_COLORS[i % BAR_COLORS.length]
                }`}
                style={{ height: `${Math.max(hPct, 2)}%` }}
              />
              {isNearest && (
                <span className="mt-0.5 text-[8px] text-amber-500">&#9660;</span>
              )}
            </div>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-slate-400">
        <span>0</span>
        <span>{nearest !== undefined ? `\u2605 nearest: C${nearest}` : ""}</span>
        <span>{sizes.length - 1}</span>
      </div>
    </div>
  );
}
