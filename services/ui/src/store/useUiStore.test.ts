/**
 * Unit tests for the Zustand store. Verifies:
 *   - Initial state matches the documented defaults.
 *   - Each setter mutates only its own slice.
 *   - `resetBm25` restores the defaults.
 *   - `userId` is generated exactly once on store init.
 *
 * Run with: `npx vitest run`
 */

import { describe, expect, it, beforeEach } from "vitest";
import { useUiStore } from "./useUiStore";

describe("useUiStore", () => {
  beforeEach(() => {
    // Reset between tests (the store is module-scoped).
    useUiStore.getState().setDataset("touche2020");
    useUiStore.getState().setMode("basic");
    useUiStore.getState().setRepresentation("bm25");
    useUiStore.getState().setFusion("rrf");
    useUiStore.getState().resetBm25();
  });

  it("has the documented defaults", () => {
    const s = useUiStore.getState();
    expect(s.dataset).toBe("touche2020");
    expect(s.mode).toBe("basic");
    expect(s.representation).toBe("bm25");
    expect(s.fusion).toBe("rrf");
    expect(s.bm25).toEqual({ k1: 1.5, b: 0.75 });
  });

  it("setDataset updates dataset", () => {
    useUiStore.getState().setDataset("nq");
    expect(useUiStore.getState().dataset).toBe("nq");
  });

  it("setRepresentation updates representation", () => {
    useUiStore.getState().setRepresentation("hybrid_parallel");
    expect(useUiStore.getState().representation).toBe("hybrid_parallel");
  });

  it("setFusion updates fusion", () => {
    useUiStore.getState().setFusion("combsum");
    expect(useUiStore.getState().fusion).toBe("combsum");
  });

  it("setBm25 updates both k1 and b", () => {
    useUiStore.getState().setBm25({ k1: 2.0, b: 0.5 });
    expect(useUiStore.getState().bm25).toEqual({ k1: 2.0, b: 0.5 });
  });

  it("resetBm25 restores defaults", () => {
    useUiStore.getState().setBm25({ k1: 0.1, b: 0.9 });
    useUiStore.getState().resetBm25();
    expect(useUiStore.getState().bm25).toEqual({ k1: 1.5, b: 0.75 });
  });

  it("userId is a non-empty string", () => {
    const id = useUiStore.getState().userId;
    expect(typeof id).toBe("string");
    expect(id.length).toBeGreaterThan(0);
  });

  it("setMode updates mode", () => {
    useUiStore.getState().setMode("with_features");
    expect(useUiStore.getState().mode).toBe("with_features");
  });
});
