/**
 * `useDatasets` — fetches the canonical dataset list from the
 * gateway's `GET /api/datasets`. The list is small (2 entries) and
 * changes only when the backend ships a new dataset, so we cache it
 * for 1 hour.
 */

import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { listDatasets } from "../api/client";
import type { ApiError, DatasetId } from "../types/api";

export function useDatasets(): UseQueryResult<DatasetId[], ApiError> {
  return useQuery<DatasetId[], ApiError>({
    queryKey: ["datasets"] as const,
    queryFn: () => listDatasets(),
    staleTime: 60 * 60 * 1000,
  });
}
