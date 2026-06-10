import { useState } from "react";

interface Turn {
  role: "user" | "assistant";
  text: string;
}

interface Props {
  turns: Turn[];
}

export default function ConversationHistory({ turns }: Props) {
  const [collapsed, setCollapsed] = useState(true);
  if (turns.length === 0) return null;

  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs dark:border-slate-600 dark:bg-slate-700">
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="mb-1 text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-200"
      >
        {collapsed ? "Show history" : "Hide history"} ({turns.length} turn{turns.length !== 1 ? "s" : ""})
      </button>
      {!collapsed && (
        <div className="space-y-1">
          {turns.map((turn, i) => (
            <p key={i} className={turn.role === "user" ? "text-slate-600 dark:text-slate-300" : "text-slate-800 dark:text-slate-100"}>
              <span className="font-semibold">{turn.role === "user" ? "Q:" : "A:"}</span> {turn.text}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
