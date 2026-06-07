/**
 * Utility for highlighting query terms inside result snippets.
 * Pure function — no React dependency. Used by `ResultCard`.
 *
 * Splits the snippet on whitespace, lowercases each token for case-
 * insensitive matching, wraps matches in `<mark>` tags, and returns
 * a `ReactNode` array of strings + `<mark>` elements.
 */

import { Fragment, type ReactNode } from "react";

/**
 * Wrap matching terms in `<mark>` tags.
 *
 * @param text  The full text (snippet) to highlight within.
 * @param terms The terms to highlight (case-insensitive). An empty
 *              array returns the text as a single string.
 * @returns     A ReactNode array suitable for `{...}` rendering.
 */
export function highlight(text: string, terms: string[]): ReactNode {
  const cleaned = terms
    .map((t) => t.trim())
    .filter((t) => t.length > 0)
    .map((t) => t.toLowerCase());
  if (cleaned.length === 0) return text;

  // Build a single regex that matches any term as a whole word.
  // Escape regex metachars in case a term contains '.' or '$' etc.
  const escaped = cleaned.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  );
  const pattern = new RegExp(`\\b(?:${escaped.join("|")})`, "gi");
  const parts: ReactNode[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(text.slice(last, match.index));
    }
    parts.push(
      <mark key={`m-${i++}-${match.index}`} className="bg-yellow-200">
        {match[0]}
      </mark>
    );
    last = match.index + match[0].length;
    if (pattern.lastIndex === match.index) {
      // Defensive: zero-width match (shouldn't happen with \b).
      pattern.lastIndex++;
    }
  }
  if (last < text.length) {
    parts.push(text.slice(last));
  }
  return parts.map((p, idx) =>
    typeof p === "string" ? <Fragment key={`t-${idx}`}>{p}</Fragment> : p
  );
}

/**
 * Truncate a string to `max` characters, ending on a word boundary
 * and appending "…" if truncation happened. Used by `ResultCard`
 * for snippet display.
 */
export function snippet(text: string, max: number = 280): string {
  if (text.length <= max) return text;
  const truncated = text.slice(0, max);
  const lastSpace = truncated.lastIndexOf(" ");
  const cut = lastSpace > max * 0.6 ? truncated.slice(0, lastSpace) : truncated;
  return `${cut}…`;
}
