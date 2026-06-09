import { useEffect, useState } from "react";

export function useDarkMode(): [boolean, () => void] {
  const [dark, setDark] = useState(
    () => document.documentElement.classList.contains("dark"),
  );

  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle("dark", dark);
    localStorage.setItem("ir-ui-dark", String(dark));
  }, [dark]);

  useEffect(() => {
    const stored = localStorage.getItem("ir-ui-dark");
    if (stored !== null) {
      setDark(stored === "true");
    }
  }, []);

  return [dark, () => setDark((d) => !d)];
}
