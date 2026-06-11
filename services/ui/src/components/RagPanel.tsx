import { useEffect, useRef, useState } from "react";
import { ragAnswer, ragAnswerStream, errorMessage } from "../api/client";
import type { DatasetId } from "../types/api";
import CitationPopover from "./CitationPopover";

interface Turn {
  role: "user" | "assistant";
  text: string;
}

interface Props {
  query: string;
  dataset: DatasetId;
  enabled: boolean;
}

const RETRIEVERS = [
  { value: "embedding" as const, label: "Embedding", hint: "Best semantic match" },
  { value: "hybrid_parallel" as const, label: "Hybrid", hint: "Lexical + semantic" },
  { value: "bm25" as const, label: "BM25", hint: "Fast, lexical only" },
];

const CITATION_RE = /\[(\d+)\]/g;

function renderAnswerWithCitations(text: string, citations: Record<string, string>): (string | JSX.Element)[] {
  const parts: (string | JSX.Element)[] = [];
  let last = 0;
  let match: RegExpExecArray | null;
  CITATION_RE.lastIndex = 0;
  while ((match = CITATION_RE.exec(text)) !== null) {
    const num = match[1];
    const docId = citations[num];
    if (match.index > last) {
      parts.push(text.slice(last, match.index));
    }
    if (docId) {
      parts.push(<CitationPopover key={match.index} num={num} docId={docId} />);
    } else {
      parts.push(<sup key={match.index} className="text-slate-400">[{num}]</sup>);
    }
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parts.push(text.slice(last));
  }
  return parts;
}

function generateId(): string {
  return "rag-" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export default function RagPanel({ query, dataset, enabled }: Props) {
  const [open, setOpen] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [citations, setCitations] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [retriever, setRetriever] = useState<"bm25" | "embedding" | "hybrid_parallel">("embedding");
  const [refinedQuery, setRefinedQuery] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(true);
  const [stage, setStage] = useState<string | null>(null);
  const [conversationEnabled, setConversationEnabled] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [inlineQuery, setInlineQuery] = useState("");
  const conversationIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setOpen(false);
  }, [query]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [answer, turns]);

  if (!enabled) return null;

  const canAsk = query.trim().length > 0 && !loading;

  function resetState() {
    setLoading(true);
    setErr(null);
    setAnswer(null);
    setSources([]);
    setCitations({});
    setRefinedQuery(null);
    setStage(null);
  }

  async function _ask(q: string) {
    resetState();

    if (!conversationIdRef.current && conversationEnabled) {
      conversationIdRef.current = generateId();
    }

    const convId = conversationEnabled ? conversationIdRef.current : undefined;

    if (!streaming) {
      try {
        const r = await ragAnswer({ query: q, dataset_id: dataset, k: 5, max_tokens: 256, retriever, conversation_id: convId });
        setAnswer(r.answer ?? "(no answer returned)");
        setSources(r.source_doc_ids ?? []);
        setCitations(r.citations ?? {});
        setRefinedQuery(r.refined_query ?? null);
        if (convId && r.answer) {
          setTurns((prev) => [...prev, { role: "user", text: q }, { role: "assistant", text: r.answer! }]);
        }
      } catch (e) {
        setErr(errorMessage(e));
      } finally {
        setLoading(false);
      }
      return;
    }

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    await ragAnswerStream(
      { query: q, dataset_id: dataset, k: 5, max_tokens: 256, retriever, conversation_id: convId },
      {
        onStage: (s, data) => {
          setStage(s);
          if (data.source_doc_ids) setSources(data.source_doc_ids as string[]);
          if (data.refined_query) setRefinedQuery(data.refined_query as string);
        },
        onToken: (token) => {
          setAnswer((prev) => (prev ?? "") + token);
        },
        onDone: (ans, srcs, _latency, refined, citations_) => {
          setAnswer(ans);
          setSources(srcs);
          setCitations(citations_ ?? {});
          setRefinedQuery(refined);
          setLoading(false);
          setStage(null);
          if (convId && ans) {
            setTurns((prev) => [...prev, { role: "user", text: q }, { role: "assistant", text: ans }]);
          }
        },
        onError: (msg) => {
          setErr(msg);
          setLoading(false);
          setStage(null);
        },
      },
      ctrl.signal,
    );
  }

  async function onAsk() {
    setOpen(true);
    await _ask(query);
  }

  async function onInlineSend() {
    const q = inlineQuery.trim();
    if (!q || loading) return;
    setInlineQuery("");
    setOpen(true);
    await _ask(q);
  }

  const inputDisabled = loading;

  return (
    <section className="rounded-md border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-600 dark:bg-slate-800">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          RAG answer
        </h2>
        <div className="flex items-center gap-2">
          <select
            value={retriever}
            onChange={(e) => setRetriever(e.target.value as "bm25" | "embedding" | "hybrid_parallel")}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
            aria-label="Retrieval method"
          >
            {RETRIEVERS.map((r) => (
              <option key={r.value} value={r.value} title={r.hint}>
                {r.label}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
              className="h-3 w-3 rounded border-slate-300 text-indigo-600"
            />
            Stream
          </label>
          <label className="flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400">
            <input
              type="checkbox"
              checked={conversationEnabled}
              onChange={(e) => setConversationEnabled(e.target.checked)}
              className="h-3 w-3 rounded border-slate-300 text-indigo-600"
            />
            Chat
          </label>
          <button
            type="button"
            disabled={!canAsk}
            onClick={onAsk}
            className="rounded-md bg-emerald-600 px-3 py-1 text-xs font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Asking\u2026" : "Get an answer"}
          </button>
        </div>
      </div>
      {open && (
        <div className="mt-2 space-y-3 text-sm">
          {conversationEnabled && turns.length > 0 && (
            <div className="flex flex-col gap-3">
              {turns.map((turn, i) => (
                <div
                  key={i}
                  className={`rounded-md px-3 py-2 ${
                    turn.role === "user"
                      ? "ml-8 bg-indigo-50 text-indigo-900 dark:bg-indigo-900/30 dark:text-indigo-100"
                      : "mr-8 bg-slate-50 text-slate-800 dark:bg-slate-700 dark:text-slate-100"
                  }`}
                >
                  <span className="text-xs font-semibold opacity-60">
                    {turn.role === "user" ? "You" : "RAG"}
                  </span>
                  <p className="mt-0.5 text-sm">{turn.text}</p>
                </div>
              ))}
            </div>
          )}
          {refinedQuery && refinedQuery !== query && (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Expanded from: <span className="italic">{query}</span> {"\u2192"}{" "}
              <span className="font-medium">{refinedQuery}</span>
            </p>
          )}
          {stage && (
            <p className="text-xs text-slate-400 dark:text-slate-500">
              {stage === "retrieval" ? "Retrieving documents\u2026" : stage}
            </p>
          )}
          {loading && !stage && (
            <p className="text-slate-500 dark:text-slate-400">
              Generating
              <span className="typing-dot">.</span>
              <span className="typing-dot">.</span>
              <span className="typing-dot">.</span>
            </p>
          )}
          {err && (
            <p
              role="alert"
              className="rounded-md border border-amber-200 bg-amber-50 p-2 text-amber-900"
            >
              {err}
            </p>
          )}
          {answer && (
            <div
              className={`rounded-md px-3 py-2 ${
                conversationEnabled ? "mr-8 bg-slate-50 dark:bg-slate-700" : ""
              }`}
            >
              {conversationEnabled && (
                <span className="text-xs font-semibold opacity-60">RAG</span>
              )}
              <p className="mt-0.5 text-slate-800 dark:text-slate-100">
                {renderAnswerWithCitations(answer, citations)}
              </p>
              {Object.keys(citations).length > 0 && (
                <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  <span className="font-semibold">Cited sources:</span>
                  <ul className="ml-2 list-inside list-disc">
                    {Object.entries(citations).map(([num, docId]) => (
                      <li key={num}>
                        [{num}] {docId}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {Object.keys(citations).length === 0 && sources.length > 0 && (
                <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                  Sources: {sources.join(", ")}
                </p>
              )}
            </div>
          )}

          <div className="flex gap-2 border-t border-slate-200 pt-2 dark:border-slate-600">
            <input
              type="text"
              value={inlineQuery}
              onChange={(e) => setInlineQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void onInlineSend();
                }
              }}
              placeholder="Ask a follow-up question\u2026"
              disabled={inputDisabled}
              className="flex-1 rounded border border-slate-300 bg-white px-2 py-1.5 text-xs focus:border-indigo-500 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-slate-100"
              aria-label="Follow-up question"
            />
            <button
              type="button"
              onClick={onInlineSend}
              disabled={inlineQuery.trim().length === 0 || inputDisabled}
              className="rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Send
            </button>
          </div>
          <div ref={chatEndRef} />
        </div>
      )}
    </section>
  );
}
