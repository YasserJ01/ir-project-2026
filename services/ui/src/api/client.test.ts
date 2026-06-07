/**
 * Unit tests for the API client's `errorMessage` helper. Verifies
 * that the helper returns a friendly string for the 4 common error
 * shapes:
 *   - 501 (RAG stub): returns the Phase 8 placeholder.
 *   - Network error (no response): "Network error: is the gateway..."
 *   - 502 with `GatewayErrorResponse` body in `detail`: surfaces `detail`.
 *   - Generic Error: returns `err.message`.
 */

import { describe, expect, it } from "vitest";
import { errorMessage } from "./client";

describe("errorMessage", () => {
  it("returns the Phase 8 placeholder for 501", () => {
    const fake = {
      isAxiosError: true,
      response: { status: 501, data: { detail: "RAG not yet" } },
      message: "Request failed with status code 501",
      config: { url: "/rag/answer" },
    } as unknown;
    expect(errorMessage(fake)).toContain("Phase 8");
  });

  it("returns network error when no response", () => {
    const fake = {
      isAxiosError: true,
      response: undefined,
      message: "Network Error",
      config: { url: "/search" },
    } as unknown;
    expect(errorMessage(fake)).toContain("Network error");
  });

  it("surfaces gateway detail string", () => {
    const fake = {
      isAxiosError: true,
      response: { status: 502, data: { detail: "indexing unreachable" } },
      message: "x",
      config: { url: "/search" },
    } as unknown;
    expect(errorMessage(fake)).toBe("indexing unreachable");
  });

  it("returns err.message for a plain Error", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
  });

  it("returns 'Unknown error.' for non-Error throws", () => {
    expect(errorMessage("oops")).toBe("Unknown error.");
    expect(errorMessage(null)).toBe("Unknown error.");
    expect(errorMessage(42)).toBe("Unknown error.");
  });
});
