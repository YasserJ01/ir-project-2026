/**
 * `SearchBar` — the search input + button. The component is
 * controlled (parent owns `value` and `onChange`) and the home
 * page wires it to local state for fast keystrokes. The parent
 * also passes `loading` so we can show a spinner instead of the
 * magnifier.
 *
 * The Enter key triggers `onSubmit`; the button is type="submit"
 * inside a <form> so screen readers handle it correctly.
 */

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
  placeholder?: string;
}

export default function SearchBar({
  value,
  onChange,
  onSubmit,
  loading,
  placeholder = "Search…",
}: Props) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (!loading) onSubmit();
      }}
      className="flex items-stretch gap-2"
    >
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoComplete="off"
        className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
      />
      <button
        type="submit"
        disabled={loading || value.trim().length === 0}
        className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        {loading ? (
          <>
            <span
              className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"
              aria-hidden
            />
            <span>Searching…</span>
          </>
        ) : (
          <>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="h-4 w-4"
              aria-hidden
            >
              <path
                fillRule="evenodd"
                d="M9 3a6 6 0 100 12 6 6 0 000-12zM1 9a8 8 0 1114.32 4.906l5.387 5.387a1 1 0 01-1.414 1.414l-5.387-5.387A8 8 0 011 9z"
                clipRule="evenodd"
              />
            </svg>
            <span>Search</span>
          </>
        )}
      </button>
    </form>
  );
}
