import { useState } from "react";

/**
 * Phase 0 placeholder.
 * Real components are implemented in Phase 7.
 */
function App() {
  const [count, setCount] = useState(0);

  return (
    <main className="min-h-screen flex items-center justify-center">
      <div className="max-w-xl text-center space-y-4 p-8 bg-white shadow rounded-lg">
        <h1 className="text-3xl font-bold">IR Search Engine — 2026</h1>
        <p className="text-slate-600">
          Phase 0 bootstrap. Real UI lands in Phase 7.
        </p>
        <button
          onClick={() => setCount((c) => c + 1)}
          className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition"
        >
          Smoke test — count: {count}
        </button>
      </div>
    </main>
  );
}

export default App;
