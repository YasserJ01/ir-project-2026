import ErrorBoundary from "./components/ErrorBoundary";
import HomePage from "./pages/HomePage";

/**
 * Phase 7 entry point. The QueryClientProvider lives in `main.tsx`
 * (along with React.StrictMode); `App` is just a thin wrapper that
 * renders the single HomePage route.
 */
function App() {
  return (
    <ErrorBoundary>
      <HomePage />
    </ErrorBoundary>
  );
}

export default App;
