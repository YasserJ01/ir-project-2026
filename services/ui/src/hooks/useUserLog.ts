/**
 * `useUserLog` — a mutation that fires `POST /api/log/click` when a
 * user clicks a result card. The mutation is fire-and-forget: errors
 * are logged to the console but never surfaced (we don't want a
 * failed log write to block the user's click navigation).
 *
 * The hook returns the `mutate` function; components call it on
 * click. We pass the required fields (user_id, query, doc_id,
 * dataset_id) and an optional client-side timestamp.
 */

import { useMutation } from "@tanstack/react-query";
import { logClick } from "../api/client";
import type { LogClickRequest } from "../types/api";

export function useUserLog() {
  return useMutation<void, Error, LogClickRequest>({
    mutationFn: (payload) => logClick(payload),
    onError: (err) => {
      // Log-clicks must never crash the UI. Swallow + log.
      console.warn("[useUserLog] click log failed:", err.message);
    },
  });
}
