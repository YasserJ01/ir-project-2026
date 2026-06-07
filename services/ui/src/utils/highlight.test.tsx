/**
 * Unit tests for the snippet/highlight utility.
 *   - snippet(): truncates on a word boundary and appends "…".
 *   - highlight(): returns a React fragment whose rendered HTML
 *     contains <mark> elements at the matching offsets.
 */

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { snippet, highlight } from "./highlight";

describe("snippet", () => {
  it("returns the text unchanged if shorter than max", () => {
    const text = "short text";
    expect(snippet(text, 100)).toBe(text);
  });

  it("truncates on a word boundary and appends …", () => {
    const text =
      "The quick brown fox jumps over the lazy dog and runs into the forest.";
    const out = snippet(text, 30);
    expect(out.length).toBeLessThanOrEqual(31);
    expect(out.endsWith("…")).toBe(true);
    // Last "word" before the ellipsis should be a complete word
    // (i.e., followed by a space before the original cut).
    expect(out).toMatch(/\b\w+…$/);
  });
});

describe("highlight", () => {
  it("returns the plain text when no terms", () => {
    const text = "hello world";
    expect(highlight(text, [])).toBe(text);
  });

  it("wraps at least one matching term in a <mark> element", () => {
    const out = highlight("the quick brown fox", ["quick"]);
    const html = renderToStaticMarkup(<>{out}</>);
    expect(html).toMatch(/<mark[^>]*>quick<\/mark>/i);
  });

  it("is case-insensitive for both the search and the term", () => {
    const out = highlight("The Quick Brown Fox", ["quick", "FOX"]);
    const html = renderToStaticMarkup(<>{out}</>);
    expect(html).toMatch(/<mark[^>]*>Quick<\/mark>/);
    expect(html).toMatch(/<mark[^>]*>Fox<\/mark>/);
  });
});
