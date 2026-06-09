/**
 * `useSearch` — React Query wrapper around the unified `/search`
 * gateway endpoint. The component layer doesn't talk to axios
 * directly; it goes through this hook.
 *
 * The query is keyed on the full SearchRequest (so changing any
 * param re-fetches). The query is **disabled** until `query` is
 * non-empty — empty queries are not legal per the gateway's
 * `GatewaySearchRequest` schema (Pydantic 422s on empty `query`).
 *
 * `staleTime: 30s` means rapid slider changes don't refetch every
 * keystroke (the debouncing happens in the slider component too,
 * so this is belt-and-suspenders).
 */

import { keepPreviousData, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { search, errorMessage } from "../api/client";
import type {
  ApiError,
  SearchRequest,
  SearchResponse,
} from "../types/api";

export function useSearch(
  req: SearchRequest
): UseQueryResult<SearchResponse, ApiError> {
  return useQuery<SearchResponse, ApiError>({
    queryKey: ["search", req] as const,
    enabled: req.query.trim().length > 0,
    queryFn: () => search(req),
    staleTime: 30_000,
    retry: 1,
    placeholderData: keepPreviousData,
  });
}

/** Re-export the helper so component code only needs one import. */
export { errorMessage };
