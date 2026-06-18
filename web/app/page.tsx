'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import EvalDashboard from './components/EvalDashboard';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type Phase = 'idle' | 'querying' | 'clarifying' | 'researching' | 'done' | 'error';
type LogType = 'start' | 'plan' | 'subtask' | 'debate' | 'synthesis' | 'report' | 'complete' | 'clarify' | 'error';
type LibrarySource    = { run_id: string; title: string; query: string };
type LibraryStepType  = 'searching' | 'chunks_retrieved' | 'generating' | 'done' | 'error';
interface LibraryStep  { id: number; type: LibraryStepType; label: string; detail?: string; ts: string; }
interface RetrievedChunk { content: string; title: string; run_id: string; }

interface SubtaskState { question: string; status: 'pending' | 'done'; findingsCount: number; }
interface ChatMessage  { role: 'user' | 'assistant'; content: string; }
interface LibraryChatMessage { role: 'user' | 'assistant'; content: string; sources?: LibrarySource[]; }
interface LogEntry     { id: number; type: LogType; label: string; detail?: string; ts: string; createdAt: number; serverTs?: number; }
interface ModelOption  { id: string; label: string; description: string; }
interface DebateTurn   { agent: 'advocate' | 'skeptic'; model: string; round: number; content: string; }
// One judged category in the debate verdict table
interface VerdictRow  { category: string; assessment: string; winner: 'proposition' | 'opposition' | 'draw'; }
// The neutral lead model's judgment of the finished debate
interface DebateVerdict { rows: VerdictRow[]; winner: 'proposition' | 'opposition' | 'draw'; model: string; }

interface UsageStats {
  leadModel: string;
  subagentModel: string;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  totalTokens: number;
  costUsd: number;
  elapsedSeconds: number;
}

interface HistoryEntry {
  id: string;
  query: string;
  title: string;
  runId: string;
  createdAt: number;
  phase: Phase;
  subtasks: SubtaskState[];
  sources: string[];
  log: LogEntry[];
  report: string;
  showReport: boolean;
  chatMessages: ChatMessage[];
  usageStats: UsageStats | null;
  debateTurns?: DebateTurn[];
  // Second research round: gap questions distilled from the debate
  gapSubtasks?: SubtaskState[];
  // Whether this run was started in debate mode (drives milestones/progress)
  debateRun?: boolean;
  debateVerdict?: DebateVerdict | null;
}

const HISTORY_KEY = 'dra_history_v1';
const ACTIVE_ID_KEY = 'dra_active_id_v1';
const CLIENT_ID_KEY = 'dra_client_id_v1';

// Clickable starter queries on the New Research page — click to research immediately
const SUGGESTED_QUERIES = [
  'Should governments regulate frontier AI models?',
  'Will AI replace more software engineers than it creates by 2030?',
  // 'Can LLM-based sentiment analysis consistently outperform traditional financial indicators?',
  // 'Should companies adopt AI agents instead of SaaS workflows?',
  'Is open-source AI more beneficial to society than proprietary AI?',
  'Will AGI arrive before 2030?',
  'Is a Computer Science degree still worth it in the AI era?',
  // 'Can solo founders build billion-dollar companies using AI?',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function readStream(
  response: Response,
  onEvent: (data: Record<string, unknown>) => void,
): Promise<void> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try { onEvent(JSON.parse(line.slice(6))); } catch { /* skip */ }
      }
    }
  }
}

function getDomain(url: string) {
  try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return url; }
}

function nowTs() {
  return new Date().toLocaleTimeString('en-US', { hour12: true, hour: 'numeric', minute: '2-digit', second: '2-digit' });
}

let _lid = 0;
function mkLog(type: LogType, label: string, detail?: string, serverTs?: number): LogEntry {
  return { id: ++_lid, type, label, detail, ts: nowTs(), createdAt: Date.now(), serverTs };
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-blue-500 flex-shrink-0" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

function MenuIcon({ className = 'w-5 h-5' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18M3 12h18M3 18h18" />
    </svg>
  );
}

function SendIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 2L11 13" /><path d="M22 2L15 22L11 13L2 9L22 2Z" />
    </svg>
  );
}

function InfoIcon({ className = 'w-3.5 h-3.5' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Model picker (custom dropdown — replaces the native <select>)
// ---------------------------------------------------------------------------

function ModelPicker({
  options, value, onChange, disabled,
}: {
  options: ModelOption[];
  value: string;
  onChange: (id: string) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onPointerDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onPointerDown);
    return () => document.removeEventListener('mousedown', onPointerDown);
  }, []);

  const selected = options.find(o => o.id === value);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        disabled={disabled}
        className="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg px-2.5 py-1.5 transition-colors disabled:cursor-default disabled:opacity-50 focus:outline-none"
      >
        {selected?.label ?? 'Model'}
        <svg className={`w-3 h-3 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute top-full right-0 mt-2 w-64 max-w-[80vw] rounded-xl border border-gray-200 bg-white shadow-lg overflow-hidden z-10">
          {options.map(opt => {
            const isSelected = opt.id === value;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => { onChange(opt.id); setOpen(false); }}
                className={`flex w-full items-start gap-2 px-3.5 py-2.5 text-left transition-colors focus:outline-none ${
                  isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${isSelected ? 'text-blue-700' : 'text-gray-800'}`}>{opt.label}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{opt.description}</p>
                </div>
                {isSelected && (
                  <svg className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StepIconCircle({ type }: { type: LogType }) {
  type Cfg = { bg: string; color: string; d: string };
  const cfg: Record<LogType, Cfg> = {
    start:    { bg: 'bg-blue-100',   color: 'text-blue-600',   d: 'M21 21l-4.35-4.35M17 11A6 6 0 111 11a6 6 0 0116 0z' },
    plan:     { bg: 'bg-purple-100', color: 'text-purple-600', d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2' },
    subtask:  { bg: 'bg-green-100',  color: 'text-green-600',  d: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
    debate:   { bg: 'bg-rose-100',   color: 'text-rose-600',   d: 'M12 3v18M7 6.5L4 13a3 3 0 006 0L7 6.5zm10 0L14 13a3 3 0 006 0l-3-6.5zM5 6.5h14' },
    synthesis:{ bg: 'bg-orange-100', color: 'text-orange-600', d: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' },
    report:   { bg: 'bg-indigo-100', color: 'text-indigo-600', d: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
    complete: { bg: 'bg-emerald-100',color: 'text-emerald-600',d: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' },
    clarify:  { bg: 'bg-yellow-100', color: 'text-yellow-600', d: 'M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' },
    error:    { bg: 'bg-red-100',    color: 'text-red-600',    d: 'M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z' },
  };
  const c = cfg[type];
  return (
    <div className={`w-8 h-8 rounded-full ${c.bg} flex items-center justify-center flex-shrink-0`}>
      <svg className={`w-4 h-4 ${c.color}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d={c.d} />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Research question cards — used for both the initial plan and the follow-up
// (gap) round. Collapsible like the debate panel so finished rounds stay
// re-openable after the report; cards keep full size while any question is
// still being researched, then all shrink together once the round finishes.
// ---------------------------------------------------------------------------

function SubtaskCards({
  heading, icon, subtasks, expanded, onToggle,
}: {
  heading: string;
  icon: string;
  subtasks: SubtaskState[];
  expanded: boolean;
  onToggle: () => void;
}) {
  const allDone = subtasks.length > 0 && subtasks.every(s => s.status === 'done');
  const totalFindings = subtasks.reduce((n, s) => n + s.findingsCount, 0);
  return (
    <div className="px-4 sm:px-6 lg:px-8 pt-4">
      <div className="border border-gray-200 rounded-2xl overflow-hidden">
        <button
          onClick={onToggle}
          className="w-full bg-gray-50 px-5 py-3 flex items-center gap-2 text-left focus:outline-none"
        >
          <span className="text-base">{icon}</span>
          <span className="text-sm font-semibold text-gray-800">
            {heading} ({subtasks.length} question{subtasks.length !== 1 ? 's' : ''})
          </span>
          {allDone && totalFindings > 0 && (
            <span className="text-[11px] text-emerald-600 font-medium bg-emerald-50 border border-emerald-200 rounded-full px-2 py-0.5">
              {totalFindings} finding{totalFindings !== 1 ? 's' : ''}
            </span>
          )}
          <svg
            className={`w-4 h-4 text-gray-400 ml-auto flex-shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {expanded && (
          <div className="px-3 sm:px-4 py-4 space-y-2">
            {subtasks.map((s, i) => (
              <div
                key={i}
                className={`flex items-start gap-3 rounded-xl border transition-all ${
                  s.status === 'done' ? 'bg-emerald-50 border-emerald-200' : 'bg-gray-50 border-gray-200'
                } ${allDone ? 'px-3 py-2' : 'px-4 py-3'}`}
              >
                {s.status === 'done'
                  ? <span className="mt-1.5 w-2 h-2 rounded-full flex-shrink-0 bg-emerald-500" />
                  : <span className="mt-0.5 flex-shrink-0"><Spinner /></span>
                }
                <span className={`flex-1 leading-relaxed ${
                  s.status === 'done' ? 'text-emerald-800' : 'text-gray-600'
                } ${allDone ? 'text-xs' : 'text-sm'}`}>
                  <span className={`font-mono font-semibold mr-1.5 ${
                    s.status === 'done' ? 'text-emerald-500' : 'text-gray-400'
                  }`}>
                    Subagent {i + 1}:
                  </span>
                  {s.question.charAt(0).toUpperCase() + s.question.slice(1)}
                </span>
                {s.status === 'done' && (
                  <span className="text-[11px] text-emerald-600 font-medium whitespace-nowrap">
                    {s.findingsCount} finding{s.findingsCount !== 1 ? 's' : ''}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Debate chat bubble — proposition speaks from the left, opposition from the
// right, like two AIs in a group chat. `streaming` renders the in-flight turn
// with a cursor.
// ---------------------------------------------------------------------------

// Display label for a verdict-table winner cell ("proposition" | "opposition" | "draw")
function verdictWinnerLabel(winner: VerdictRow['winner']) {
  return winner === 'draw' ? 'Draw' : winner === 'proposition' ? 'Proposition' : 'Opposition';
}

function DebateBubble({
  agent, round, modelLabel, content, streaming = false, thinking = false,
}: {
  agent: DebateTurn['agent'];
  round: number;
  modelLabel: string;
  content: string;
  streaming?: boolean;
  thinking?: boolean;
}) {
  const isSkeptic = agent === 'skeptic';
  return (
    <div className={`flex items-start gap-2 ${isSkeptic ? 'flex-row-reverse' : ''}`}>
      <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold flex-shrink-0 ${
        isSkeptic ? 'bg-amber-100 text-amber-700' : 'bg-blue-100 text-blue-700'
      }`}>
        {isSkeptic ? 'O' : 'P'}
      </div>
      <div className={`max-w-[88%] sm:max-w-[80%] rounded-2xl border px-4 py-3 ${
        isSkeptic
          ? 'bg-amber-50 border-amber-100 rounded-br-md'
          : 'bg-blue-50 border-blue-100 rounded-bl-md'
      }`}>
        <div className={`flex items-baseline gap-2 mb-1.5 ${isSkeptic ? 'justify-end' : ''}`}>
          <span className={`text-[11px] font-semibold ${isSkeptic ? 'text-amber-700' : 'text-blue-700'}`}>
            {isSkeptic ? 'Opposition' : 'Proposition'}
          </span>
          <span className="text-[11px] text-gray-400">
            {thinking
              ? `${modelLabel} · thinking…`
              : `Round ${round} · ${modelLabel}${streaming ? ' · typing…' : ''}`}
          </span>
        </div>
        {thinking ? (
          <div className="flex items-center gap-1 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce [animation-delay:-0.3s]" />
            <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce [animation-delay:-0.15s]" />
            <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" />
          </div>
        ) : (
          <div className="text-sm text-gray-700 leading-relaxed [&_p]:mb-2 [&_p:last-child]:mb-0
            [&_strong]:font-semibold [&_strong]:text-gray-900
            [&_a]:text-blue-600 [&_a:hover]:underline [&_a]:break-words
            [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:mb-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:mb-2">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            {streaming && (
              <span className="inline-block w-1.5 h-3.5 bg-gray-400 animate-pulse rounded-sm" />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sidebar (New Research + Recents history)
// ---------------------------------------------------------------------------

function HistoryItemIcon() {
  return (
    <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.86 9.86 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6M4 7h16M9 7V4a1 1 0 011-1h4a1 1 0 011 1v3" />
    </svg>
  );
}

function Sidebar({
  history, activeId, locked, mobileOpen, view, publicRuns, onNewResearch, onShowEvalDashboard, onShowLibrary, onSelect, onDelete, onPublicSelect, onClose,
}: {
  history: HistoryEntry[];        // localStorage entries for cache lookup + delete + live dot
  activeId: string | null;
  locked: boolean;
  mobileOpen: boolean;
  view: 'research' | 'eval' | 'library';
  publicRuns: {id: string; title: string; query: string}[];
  onNewResearch: () => void;
  onShowEvalDashboard: () => void;
  onShowLibrary: () => void;
  onSelect: (entry: HistoryEntry) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
  onPublicSelect: (runId: string) => void;
  onClose: () => void;
}) {
  const evalActive = view === 'eval';
  const libraryActive = view === 'library';
  const newResearchActive = view === 'research' && activeId === null;
  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-60 flex-shrink-0 bg-gray-50 border-r border-gray-200 flex flex-col select-none
          transform transition-transform duration-200 ease-in-out
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}
          lg:static lg:translate-x-0 lg:z-auto`}
      >
      {/* Brand */}
      <div className="px-5 py-5 border-b border-gray-200 flex items-center justify-between">
        <h1 className="text-[20px] font-extrabold text-gray-900 tracking-tight leading-tight">
          MindClash
          <span className="block text-[12px] font-bold text-gray-400 tracking-normal mt-2">
            Multi-agent deep research with
            <br />
            debate system
          </span>
     
        </h1>
        <button
          onClick={onClose}
          className="lg:hidden flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
          aria-label="Close menu"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* New Research */}
      <div className="px-3 pt-4 space-y-1.5">
        <button
          onClick={onNewResearch}
          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl border font-semibold transition-colors text-sm ${
            newResearchActive
              ? 'bg-white border-gray-200 text-gray-900 shadow-sm'
              : 'bg-transparent border-transparent text-gray-500 hover:bg-gray-100 hover:text-gray-800'
          }`}
        >
          <svg className="w-[18px] h-[18px] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
          </svg>
          New Research
        </button>

        {/* Research Library */}
        <button
          onClick={onShowLibrary}
          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl border font-semibold transition-colors text-sm ${
            libraryActive
              ? 'bg-white border-gray-200 text-gray-900 shadow-sm'
              : 'bg-transparent border-transparent text-gray-500 hover:bg-gray-100 hover:text-gray-800'
          }`}
        >
          <svg className="w-[18px] h-[18px] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
          </svg>
          Research Library
        </button>

        {/* Eval Dashboard */}
        <button
          onClick={onShowEvalDashboard}
          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl border font-semibold transition-colors text-sm ${
            evalActive
              ? 'bg-white border-gray-200 text-gray-900 shadow-sm'
              : 'bg-transparent border-transparent text-gray-500 hover:bg-gray-100 hover:text-gray-800'
          }`}
        >
          <svg className="w-[18px] h-[18px] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 3v18h18M8 17V11M13 17V7M18 17v-4" />
          </svg>
          Eval Dashboard
        </button>
      </div>

      {/* History — unified public feed; local entries load from cache, others fetch from API */}
      <div className="flex-1 min-h-0 flex flex-col px-3 pt-5 pb-3">
        <p className="px-3 text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          History
        </p>
        <div className="flex-1 overflow-y-auto space-y-0.5 text-[13px]">
          {publicRuns.length === 0 ? (
            <p className="px-3 py-2 text-xs text-gray-400">No research yet</p>
          ) : publicRuns.map(run => {
            const localEntry = history.find(h => h.runId === run.id);
            // activeId is a localStorage UUID for local entries, or the database run ID
            // for public-only runs — check both so local entries highlight correctly.
            const isActive = view === 'research' && (run.id === activeId || (localEntry != null && localEntry.id === activeId));
            const isLive = !!localEntry && isActive && (localEntry.phase === 'researching' || localEntry.phase === 'querying' || localEntry.phase === 'clarifying');
            const isDisabled = locked && !isActive;
            return (
              <div
                key={run.id}
                role="button"
                tabIndex={0}
                aria-disabled={isDisabled}
                onClick={() => { if (isDisabled) return; localEntry ? onSelect(localEntry) : onPublicSelect(run.id); }}
                onKeyDown={e => { if (e.key === 'Enter' && !isDisabled) { localEntry ? onSelect(localEntry) : onPublicSelect(run.id); } }}
                className={`group w-full flex items-center gap-2 px-3 py-2 rounded-xl transition-colors ${
                  isActive
                    ? 'bg-white border border-gray-200 text-gray-900 font-semibold shadow-sm'
                    : 'text-gray-500 hover:bg-gray-100 hover:text-gray-800'
                } ${isDisabled ? 'opacity-50 cursor-default' : 'cursor-pointer'}`}
              >
                <span className="flex-1 truncate">{run.title || run.query || 'Untitled'}</span>
                {isLive && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse flex-shrink-0" />}
                {localEntry && !isLive && (
                  <button
                    onClick={e => onDelete(localEntry.id, e)}
                    className={`flex-shrink-0 text-gray-400 hover:text-red-500 transition-opacity p-0.5 ${isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                    aria-label="Delete research"
                  >
                    <TrashIcon />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </div>
      </aside>
    </>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Home() {
  const [phase, setPhase]     = useState<Phase>('idle');
  const [query, setQuery]     = useState('');
  const [title, setTitle]     = useState('');
  const [runId, setRunId]     = useState('');
  const [error, setError]     = useState('');
  const [report, setReport]   = useState('');
  const [showReport, setShowReport] = useState(false);

  const [clarifyQuestions, setClarifyQuestions] = useState<string[]>([]);
  const [clarifyOptions,   setClarifyOptions]   = useState<string[][]>([]);
  const [clarifyAnswers,   setClarifyAnswers]   = useState<string[]>([]);

  const [subtasks, setSubtasks] = useState<SubtaskState[]>([]);
  const [sources,  setSources]  = useState<string[]>([]);
  const [log,      setLog]      = useState<LogEntry[]>([]);

  const [chatMessages,  setChatMessages]  = useState<ChatMessage[]>([]);
  const [chatInput,     setChatInput]     = useState('');
  const [chatStreaming, setChatStreaming] = useState(false);

  const [rightTab,      setRightTab]     = useState<'steps' | 'sources'>('steps');
  const [collapsedLogs, setCollapsedLogs] = useState<Set<number>>(new Set());

  const [supervisorThinking,         setSupervisorThinking]         = useState('');
  const [supervisorThinkingExpanded, setSupervisorThinkingExpanded] = useState(false);
  const [synthesizingActive,         setSynthesizingActive]         = useState(false);
  const [researchEndTime,            setResearchEndTime]            = useState<number | null>(null);
  const [copied,                     setCopied]                     = useState(false);
  const [usageStats,                 setUsageStats]                 = useState<UsageStats | null>(null);

  const [history,  setHistory]  = useState<HistoryEntry[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [publicRuns, setPublicRuns] = useState<{id: string; title: string; query: string}[]>([]);

  const LIBRARY_HISTORY_KEY = 'mindclash_library_history';
  const [libraryChatMessages,  setLibraryChatMessages]  = useState<LibraryChatMessage[]>(() => {
    try {
      const stored = localStorage.getItem('mindclash_library_history');
      return stored ? JSON.parse(stored) : [];
    } catch { return []; }
  });
  const [libraryChatInput,     setLibraryChatInput]     = useState('');
  const [libraryChatStreaming, setLibraryChatStreaming] = useState(false);
  const [librarySteps,         setLibrarySteps]         = useState<LibraryStep[]>([]);
  const [libraryChunks,        setLibraryChunks]        = useState<RetrievedChunk[]>([]);
  const [libraryRightTab,      setLibraryRightTab]      = useState<'steps' | 'chunks'>('steps');
  const [showLibraryActivityMobile, setShowLibraryActivityMobile] = useState(false);

  // Top-level view: research flow (default) vs eval dashboard vs RAG library
  const [view, setView] = useState<'research' | 'eval' | 'library'>('research');

  // Mobile-only UI state: off-canvas sidebar drawer + collapsible activity panel
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showActivityMobile, setShowActivityMobile] = useState(false);

  // Mirrors engine/models.py LEAD_MODEL_OPTIONS — lets the picker render
  // immediately, before the (possibly cold-starting) backend responds.
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    { id: 'gpt-5.4', label: 'GPT-5.4', description: 'Best for complex topics' },
    { id: 'gpt-5.4-mini', label: 'GPT-5.4 Mini', description: 'Faster and cheaper' },
  ]);
  const [selectedModel, setSelectedModel] = useState('gpt-5.4');

  // Debate mode: two cross-provider agents argue over the findings pre-synthesis
  const [debateMode,     setDebateMode]     = useState(false);
  const [debateInfoOpen, setDebateInfoOpen] = useState(false);
  const debateInfoRef = useRef<HTMLDivElement>(null);
  const [advocateModel,  setAdvocateModel]  = useState('gpt-5.4');
  const [skepticModel,   setSkepticModel]   = useState('gpt-5.4');
  const [debateTurns,    setDebateTurns]    = useState<DebateTurn[]>([]);
  const [debateExpanded, setDebateExpanded] = useState(true);
  const [debatingActive, setDebatingActive] = useState(false);
  // In-flight debate turn, accumulated token-by-token from debate_token events;
  // replaced by the final debate_turn event when the speaker finishes.
  const [debateStreaming, setDebateStreaming] = useState<{ agent: DebateTurn['agent']; content: string } | null>(null);
  // Whether the *active* run was started in debate mode — drives the milestone
  // labels and progress weighting (the debateMode toggle is just the form input)
  const [debateRun,      setDebateRun]      = useState(false);
  // Judge phase: the neutral lead weighs both sides after the final round
  const [judgingActive,  setJudgingActive]  = useState(false);
  const [debateVerdict,  setDebateVerdict]  = useState<DebateVerdict | null>(null);
  // Verdict card is collapsible — expanded while live, auto-collapsed once the
  // final report is ready so the report takes center stage (still re-openable)
  const [verdictExpanded, setVerdictExpanded] = useState(true);

  // Debate-driven gap research (second finding round, debate mode only)
  const [gapSubtasks,       setGapSubtasks]       = useState<SubtaskState[]>([]);
  const [gapPlanningActive, setGapPlanningActive] = useState(false);

  // Collapsible question-card rounds (like the debate panel): open while their
  // round is live, auto-collapsed when the report takes the stage
  const [planCardsExpanded, setPlanCardsExpanded] = useState(true);
  const [gapCardsExpanded,  setGapCardsExpanded]  = useState(true);

  // DOCX export in flight (the docx lib is dynamically imported + packs async)
  const [docxBusy, setDocxBusy] = useState(false);

  // Per-round question counters (refs: handleEvent runs outside React renders,
  // so state would be stale). When a round's last question completes we log a
  // "Compacting Findings" step, so "Research Execution" stops reading as running
  // while every question card already shows done.
  const stageProgress = useRef({ plan: { total: 0, done: 0 }, gap: { total: 0, done: 0 } });

  // Mirrors of subtasks/gapSubtasks for handleEvent (same staleness issue as
  // stageProgress above) — lets subtask_done look up "Subagent N" by question.
  const subtasksRef = useRef<SubtaskState[]>([]);
  const gapSubtasksRef = useRef<SubtaskState[]>([]);
  const streamingRunIdRef = useRef<string>('');

  // Anonymous per-visitor id (no login) — scopes /runs and /eval/reports so
  // visitors only see their own data. Generated once and persisted locally.
  const [clientId, setClientId] = useState('');

  const logEndRef         = useRef<HTMLDivElement>(null);
  const chatEndRef        = useRef<HTMLDivElement>(null);
  const debateEndRef      = useRef<HTMLDivElement>(null);
  const centerRef         = useRef<HTMLDivElement>(null);
  const libraryChatEndRef  = useRef<HTMLDivElement>(null);
  const libraryStepsEndRef = useRef<HTMLDivElement>(null);
  const libraryStepIdRef   = useRef(0);

  // Persist library chat history to localStorage
  useEffect(() => {
    try { localStorage.setItem(LIBRARY_HISTORY_KEY, JSON.stringify(libraryChatMessages)); } catch {}
  }, [libraryChatMessages]);

  // Follow new steps only while a run is live — opening a finished session from
  // history must show it from the top, not auto-scrolled to the bottom.
  useEffect(() => {
    if (phase === 'researching' || phase === 'querying' || phase === 'clarifying') {
      logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [log, phase]);
  // Same idea for chat: only follow messages the user is actively streaming
  useEffect(() => {
    if (chatStreaming) chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, chatStreaming]);

  // Library chat: scroll to bottom while streaming
  useEffect(() => {
    if (libraryChatStreaming) libraryChatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [libraryChatMessages, libraryChatStreaming]);

  // Library steps: always scroll to newest step
  useEffect(() => {
    if (libraryChatStreaming) libraryStepsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [librarySteps, libraryChatStreaming]);

  // Switching sessions (or pages) lands at the top of the restored session
  useEffect(() => { centerRef.current?.scrollTo({ top: 0 }); }, [activeId, view]);

  // Close the "what is debate mode" popover on outside click
  useEffect(() => {
    if (!debateInfoOpen) return;
    function onClick(e: MouseEvent) {
      if (debateInfoRef.current && !debateInfoRef.current.contains(e.target as Node)) {
        setDebateInfoOpen(false);
      }
    }
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [debateInfoOpen]);

  // Follow the live debate: keep the newest bubble (or the streaming one) in
  // view while turns arrive. Only during the run — never when a finished
  // debate is re-opened later.
  useEffect(() => {
    if (phase === 'researching' && (debatingActive || debateStreaming)) {
      debateEndRef.current?.scrollIntoView({ block: 'nearest' });
    }
  }, [phase, debatingActive, debateStreaming, debateTurns]);

  // Auto-hide the verdict card once the final report is ready, so the report
  // takes center stage. Still re-openable via the card's header toggle.
  useEffect(() => {
    if (report) setVerdictExpanded(false);
  }, [report]);

  // 1s tick while researching — drives the live elapsed time on the last
  // thinking step so it never sits on a static "Running…"
  const [nowTick, setNowTick] = useState(() => Date.now());
  useEffect(() => {
    if (phase !== 'researching') return;
    const id = setInterval(() => setNowTick(Date.now()), 1000);
    return () => clearInterval(id);
  }, [phase]);

  // Load a history entry's saved state into the active session view
  function restoreEntry(entry: HistoryEntry) {
    setQuery(entry.query);
    setTitle(entry.title || '');
    setRunId(entry.runId);
    setSubtasks(entry.subtasks);
    setSources(entry.sources);
    setLog(entry.log);
    setReport(entry.report);
    setShowReport(entry.showReport);
    setChatMessages(entry.chatMessages);
    setCollapsedLogs(new Set());
    setSupervisorThinking(''); setSupervisorThinkingExpanded(false);
    setSynthesizingActive(false); setCopied(false);
    setUsageStats(entry.usageStats);
    setDebateTurns(entry.debateTurns ?? []);  // old localStorage entries lack this
    setGapSubtasks(entry.gapSubtasks ?? []);
    setDebatingActive(false); setDebateExpanded(false); setDebateStreaming(null);
    setGapPlanningActive(false); setJudgingActive(false);
    setDebateRun(entry.debateRun ?? (entry.debateTurns?.length ?? 0) > 0);
    setDebateVerdict(entry.debateVerdict ?? null);
    setPlanCardsExpanded(false); setGapCardsExpanded(false);
    maxProgress.current = 0;
    setError(''); setClarifyQuestions([]); setClarifyOptions([]); setClarifyAnswers([]);
    // Transient phases can't be resumed after a refresh/switch — reset them
    const transient: Phase[] = ['researching', 'querying', 'clarifying'];
    const restoredPhase = transient.includes(entry.phase) ? (entry.report ? 'done' : 'idle') : entry.phase;
    setPhase(restoredPhase);
    // For a finished run, anchor the last step's duration to its own timestamp
    // instead of leaving it stuck on "Running…"
    const lastLog = entry.log[entry.log.length - 1];
    setResearchEndTime(restoredPhase === 'done' && lastLog ? (lastLog.serverTs ?? lastLog.createdAt) : null);
  }

  // Fetch selectable lead models for the New Research page — overrides the
  // hardcoded fallback above once the (possibly cold-starting) backend responds.
  useEffect(() => {
    fetch(`${API}/models`)
      .then(res => res.json())
      .then((data: {
        default: string;
        defaults?: { lead: string; advocate: string; skeptic: string };
        options: ModelOption[];
      }) => {
        if (data.options?.length) setModelOptions(data.options);
        if (data.default) setSelectedModel(data.default);
        // Per-role debate defaults — cross-provider when the keys are configured
        if (data.defaults?.advocate) setAdvocateModel(data.defaults.advocate);
        if (data.defaults?.skeptic) setSkepticModel(data.defaults.skeptic);
      })
      .catch(() => { /* keep the hardcoded fallback */ });
  }, []);

  // Fetch global public run history on mount for the sidebar
  useEffect(() => {
    fetch(`${API}/runs?status=done&limit=30`)
      .then(res => res.json())
      .then((data: unknown) => setPublicRuns(Array.isArray(data) ? data : []))
      .catch(() => { /* non-critical */ });
  }, []);

  // Read or generate the anonymous client id on mount
  useEffect(() => {
    try {
      let id = localStorage.getItem(CLIENT_ID_KEY);
      if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem(CLIENT_ID_KEY, id);
      }
      setClientId(id);
    } catch {
      setClientId(crypto.randomUUID());
    }
  }, []);

  // Restore history + active session from localStorage on mount
  useEffect(() => {
    try {
      const rawHistory = localStorage.getItem(HISTORY_KEY);
      const hist: HistoryEntry[] = rawHistory ? JSON.parse(rawHistory) : [];
      setHistory(hist);
      const aid = localStorage.getItem(ACTIVE_ID_KEY);
      const entry = aid ? hist.find(h => h.id === aid) : undefined;
      if (entry) {
        restoreEntry(entry);
        setActiveId(entry.id);
      }
    } catch { /* ignore parse errors */ }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist history list to localStorage whenever it changes
  useEffect(() => {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch { /* ignore quota errors */ }
  }, [history]);

  // Persist which history entry is active
  useEffect(() => {
    try {
      if (activeId) localStorage.setItem(ACTIVE_ID_KEY, activeId);
      else localStorage.removeItem(ACTIVE_ID_KEY);
    } catch { /* ignore quota errors */ }
  }, [activeId]);

  // Keep the active history entry's saved snapshot in sync with the live session
  useEffect(() => {
    if (!activeId) return;
    setHistory(prev => prev.map(h => h.id === activeId
      ? { ...h, query, title, runId, phase, subtasks, sources, log, report, showReport, chatMessages, usageStats, debateTurns, gapSubtasks, debateRun, debateVerdict }
      : h));
  }, [activeId, phase, query, title, runId, subtasks, sources, log, report, showReport, chatMessages, usageStats, debateTurns, gapSubtasks, debateRun, debateVerdict]);

  // ---------------------------------------------------------------------------
  // Progress + milestones. Stage budgets differ per mode so the first research
  // round can't push the bar to ~90% when a debate + second round still follow:
  //   plain run:  plan 0–10 · research 10–72 · synthesis 80 · report 95 · reveal 100
  //   debate run: plan 0–8 · research 8–36 · debate 38–58 · verdict 60 ·
  //               gap plan 62 · gap research 64–84 · synthesis 88 · report 95
  // ---------------------------------------------------------------------------

  // Matches the backend DEFAULT_DEBATE_ROUNDS (2 rounds × 2 speakers)
  const EXPECTED_DEBATE_TURNS = 4;

  // Monotonic clamp: stage flags can flip off between SSE events (e.g. between
  // compaction and the next stage), and the bar must never move backward.
  const maxProgress = useRef(0);

  const progressPct = (() => {
    const raw = (() => {
      if (showReport) return 100;
      if (report) return 95;
      const doneFrac = (round: SubtaskState[]) =>
        round.length === 0 ? 0 : round.filter(s => s.status === 'done').length / round.length;
      if (debateRun) {
        if (synthesizingActive) return 88;
        if (gapSubtasks.length > 0) return 64 + Math.round(doneFrac(gapSubtasks) * 20);
        if (gapPlanningActive) return 62;
        if (judgingActive || debateVerdict) return 60;
        const turns = Math.min(debateTurns.length + (debateStreaming ? 0.5 : 0), EXPECTED_DEBATE_TURNS);
        if (turns > 0 || debatingActive) return 38 + Math.round((turns / EXPECTED_DEBATE_TURNS) * 20);
        if (subtasks.length > 0) return 8 + Math.round(doneFrac(subtasks) * 28);
        return phase === 'researching' ? 3 : 0; // 3% = tiny visible sliver
      }
      if (synthesizingActive) return 80;
      if (subtasks.length > 0) return 10 + Math.round(doneFrac(subtasks) * 62);
      return phase === 'researching' ? 3 : 0;
    })();
    return Math.max(raw, maxProgress.current);
  })();
  maxProgress.current = progressPct >= 100 ? 0 : progressPct;

  // Milestone trail shown beside the progress bar — debate runs surface the
  // extra debate / follow-up research stages
  const milestones = debateRun
    ? ['Planning', 'Initial Research', 'Debate', 'Gap Research', 'Synthesizing', 'Report']
    : ['Planning', 'Research', 'Synthesizing', 'Report'];
  const milestoneIdx = (() => {
    if (report) return milestones.length - 1;
    if (synthesizingActive) return milestones.length - 2;
    if (debateRun) {
      if (gapPlanningActive || gapSubtasks.length > 0) return 3;
      if (debatingActive || judgingActive || debateStreaming || debateTurns.length > 0) return 2;
    }
    if (subtasks.length > 0) return 1;
    return 0;
  })();

  // Reflect the current step + progress in the browser tab title while a run
  // is in flight, so progress is visible even when this tab isn't focused.
  useEffect(() => {
    const currentStep = log.length > 0 ? log[log.length - 1].label : milestones[milestoneIdx];
    document.title = phase === 'researching'
      ? `${progressPct}% · ${currentStep} — MindClash`
      : 'MindClash';
  }, [phase, progressPct, milestones, milestoneIdx, log]);

  const displayQuery = title || (query ? query.charAt(0).toUpperCase() + query.slice(1) : '');

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function addLog(type: LogType, label: string, detail?: string, serverTs?: number) {
    setLog(prev => [...prev, mkLog(type, label, detail, serverTs)]);
  }

  function toggleCollapse(id: number) {
    setCollapsedLogs(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  function handleEvent(data: Record<string, unknown>) {
    const type = data.type as string;
    // Convert server unix timestamp (seconds) to ms for consistent timing
    const serverTs = data.ts ? (data.ts as number) * 1000 : undefined;
    if (type === 'started') {
      const rid = data.run_id as string;
      streamingRunIdRef.current = rid;
      setRunId(rid);
      setPublicRuns(prev => [{ id: rid, title: query, query }, ...prev.filter(r => r.id !== rid)]);
      addLog('start', 'Initialization', undefined, serverTs);
    } else if (type === 'plan_thinking') {
      setPhase('researching');
      setSupervisorThinking(data.content as string);
      addLog('plan', 'Research Planning', data.content as string, serverTs);
    } else if (type === 'plan') {
      setPhase(p => (p === 'querying' ? 'researching' : p));
      const qs = (data.subtasks as string[]) ?? [];
      const newSubtasks: SubtaskState[] = qs.map(q => ({ question: q, status: 'pending', findingsCount: 0 }));
      setSubtasks(newSubtasks);
      subtasksRef.current = newSubtasks;
      stageProgress.current.plan = { total: qs.length, done: 0 };
      const planTitle = data.title as string | undefined;
      if (planTitle) {
        setTitle(planTitle);
        setPublicRuns(prev => prev.map(r => r.id === streamingRunIdRef.current ? { ...r, title: planTitle } : r));
      }
    } else if (type === 'subtask_done') {
      const q     = data.question as string;
      const count = (data.findings_count as number) ?? 0;
      const srcs  = (data.sources as string[]) ?? [];
      const stage = data.stage === 'gap' ? 'gap' : 'plan';
      const markDone = (s: SubtaskState) => s.question === q ? { ...s, status: 'done' as const, findingsCount: count } : s;
      const list = stage === 'gap' ? gapSubtasksRef.current : subtasksRef.current;
      const idx = list.findIndex(s => s.question === q);
      if (stage === 'gap') setGapSubtasks(prev => prev.map(markDone));
      else setSubtasks(prev => prev.map(markDone));
      setSources(prev => { const set = new Set(prev); srcs.forEach(s => set.add(s)); return [...set]; });
      const subagentLabel = `${stage === 'gap' ? 'Gap Subagent' : 'Subagent'} ${idx + 1}`;
      const label = stage === 'gap' ? 'Follow-up Research' : 'Research';
      addLog('subtask', `${label} — ${subagentLabel}`, `${count} finding${count !== 1 ? 's' : ''} · ${q.slice(0, 80)}${q.length > 80 ? '…' : ''}`, serverTs);
      // Last question of the round done → compaction starts immediately, so log
      // it now: the execution step resolves the moment every card shows done.
      const sp = stageProgress.current[stage];
      sp.done += 1;
      if (sp.total > 0 && sp.done >= sp.total) {
        addLog('synthesis', 'Compacting Findings', 'All questions answered — merging findings into a compact research summary', serverTs);
      }
    } else if (type === 'debating') {
      setDebatingActive(true);
      setDebateExpanded(true);
      setPlanCardsExpanded(false); // debate takes the stage — questions stay openable
      addLog('debate', 'Adversarial Debate', 'Agents are arguing opposing positions over the findings…', serverTs);
    } else if (type === 'debate_token') {
      const agent = data.agent as DebateTurn['agent'];
      const token = data.content as string;
      setDebateStreaming(prev =>
        prev && prev.agent === agent ? { agent, content: prev.content + token } : { agent, content: token }
      );
    } else if (type === 'debate_turn') {
      const turn: DebateTurn = {
        agent: data.agent as DebateTurn['agent'],
        model: data.model as string,
        round: data.round as number,
        content: data.content as string,
      };
      setDebateStreaming(null);
      setDebateTurns(prev => [...prev, turn]);
      const label = turn.agent === 'advocate' ? 'Proposition' : 'Opposition';
      addLog('debate', `Debate — ${label} (round ${turn.round})`, turn.content.slice(0, 160) + (turn.content.length > 160 ? '…' : ''), serverTs);
    } else if (type === 'judging') {
      setDebatingActive(false);
      setJudgingActive(true);
      setDebateExpanded(false); // debate is over — collapse it so the verdict card takes the stage
      addLog('debate', 'Judging the Debate', 'A neutral judge (the lead model) is weighing both sides to declare a winner', serverTs);
    } else if (type === 'debate_verdict') {
      const winner = data.winner as DebateVerdict['winner'];
      const rows = data.rows as VerdictRow[];
      setJudgingActive(false);
      setVerdictExpanded(true);
      setDebateVerdict({ winner, rows, model: data.model as string });
      const label = winner === 'draw' ? 'Draw' : `${verdictWinnerLabel(winner)} wins`;
      const detail = rows.map(r => `${r.category}: ${r.assessment}`).join(' ');
      addLog('debate', `Debate Verdict — ${label}`, detail.slice(0, 160) + (detail.length > 160 ? '…' : ''), serverTs);
    } else if (type === 'gap_planning') {
      setDebatingActive(false);
      setJudgingActive(false);
      setGapPlanningActive(true);
      addLog('plan', 'Identifying Evidence Gaps', 'Distilling unresolved objections from the debate into follow-up research questions', serverTs);
    } else if (type === 'gap_plan') {
      const qs = (data.subtasks as string[]) ?? [];
      const newGapSubtasks: SubtaskState[] = qs.map(q => ({ question: q, status: 'pending', findingsCount: 0 }));
      setGapPlanningActive(false);
      setDebateExpanded(false); // debate is over — focus moves to the second finding round
      setVerdictExpanded(false); // follow-up research is starting — verdict card steps aside
      setGapSubtasks(newGapSubtasks);
      gapSubtasksRef.current = newGapSubtasks;
      setGapCardsExpanded(true);
      stageProgress.current.gap = { total: qs.length, done: 0 };
      addLog('plan', 'Follow-up Research Plan', `${qs.length} gap question${qs.length !== 1 ? 's' : ''} from the debate`, serverTs);
    } else if (type === 'synthesizing') {
      setDebatingActive(false);
      setJudgingActive(false);
      setGapPlanningActive(false);
      setSynthesizingActive(true);
      // Synthesis is the last step before the report — every earlier step
      // (planning, research, debate, judging, gap research) is now complete.
      setPlanCardsExpanded(false);
      setDebateExpanded(false);
      setVerdictExpanded(false);
      setGapCardsExpanded(false);
      addLog('synthesis', 'Synthesizing Findings', undefined, serverTs);
    } else if (type === 'clarification_needed') {
      const qs = (data.questions as string[]) ?? [];
      const opts = (data.options as string[][]) ?? [];
      setRunId(data.run_id as string);
      setClarifyQuestions(qs);
      setClarifyOptions(opts);
      setClarifyAnswers(new Array(qs.length).fill(''));
      setPhase('clarifying');
      addLog('clarify', 'Clarification Required', `${qs.length} question${qs.length !== 1 ? 's' : ''} to answer`, serverTs);
    } else if (type === 'report') {
      setSynthesizingActive(false);
      setReport(data.content as string);
      setShowReport(true);
      // Report takes the stage — debate + question rounds stay openable above it
      setDebateExpanded(false);
      setPlanCardsExpanded(false);
      setGapCardsExpanded(false);
      addLog('report', 'Final Report', undefined, serverTs);
    } else if (type === 'done') {
      const endTs = serverTs ?? Date.now();
      setPhase('done');
      setResearchEndTime(endTs);
      setSubtasks(prev => prev.map(s => s.status === 'pending' ? { ...s, status: 'done' } : s));
      setGapSubtasks(prev => prev.map(s => s.status === 'pending' ? { ...s, status: 'done' } : s));
      const u = data.usage as Record<string, unknown> | undefined;
      if (u) {
        setUsageStats({
          leadModel: u.lead_model as string,
          subagentModel: u.subagent_model as string,
          inputTokens: u.input_tokens as number,
          outputTokens: u.output_tokens as number,
          cachedTokens: u.cached_tokens as number,
          totalTokens: u.total_tokens as number,
          costUsd: u.cost_usd as number,
          elapsedSeconds: u.elapsed_seconds as number,
        });
      }
      addLog('complete', 'Completed', undefined, serverTs);
    } else if (type === 'error') {
      setError(data.message as string);
      setPhase('error');
      setResearchEndTime(serverTs ?? Date.now());
      addLog('error', 'Error', data.message as string, serverTs);
    }
  }

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------

  async function startResearch(presetQuery?: string) {
    const trimmed = (presetQuery ?? query).trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setPhase('querying');
    setTitle('');
    setSubtasks([]); setSources([]); setReport(''); setShowReport(false);
    setError(''); setChatMessages([]); setRunId('');
    setSupervisorThinking(''); setSupervisorThinkingExpanded(false);
    setSynthesizingActive(false); setResearchEndTime(null); setUsageStats(null);
    setDebateTurns([]); setDebatingActive(false); setDebateExpanded(true); setDebateStreaming(null);
    setDebateRun(debateMode); setJudgingActive(false); setDebateVerdict(null);
    setGapSubtasks([]); setGapPlanningActive(false);
    setPlanCardsExpanded(true); setGapCardsExpanded(true);
    maxProgress.current = 0;
    stageProgress.current = { plan: { total: 0, done: 0 }, gap: { total: 0, done: 0 } };
    setClarifyQuestions([]); setClarifyOptions([]); setClarifyAnswers([]);
    setLog([mkLog('start', 'Initialization', `Query: ${trimmed.slice(0, 120)}`)]);
    setRightTab('steps');
    setCollapsedLogs(new Set());

    // Immediately add this run to history (Recents)
    const id = crypto.randomUUID();
    setActiveId(id);
    setHistory(prev => [{
      id, query: trimmed, title: '', runId: '', createdAt: Date.now(), phase: 'querying',
      subtasks: [], sources: [], log: [], report: '', showReport: false, chatMessages: [],
      usageStats: null, debateTurns: [], gapSubtasks: [], debateRun: debateMode, debateVerdict: null,
    }, ...prev]);

    try {
      const res = await fetch(`${API}/research`, {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Client-Id': clientId },
        body: JSON.stringify({
          query: trimmed,
          model: selectedModel || undefined,
          debate: debateMode,
          advocate_model: debateMode ? advocateModel || undefined : undefined,
          skeptic_model: debateMode ? skepticModel || undefined : undefined,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      await readStream(res, handleEvent);
    } catch (e) { setError(String(e)); setPhase('error'); }
  }

  async function submitClarification() {
    if (clarifyAnswers.some(a => !a.trim())) return;
    setPhase('researching');
    addLog('clarify', 'Answers submitted');
    try {
      const res = await fetch(`${API}/runs/${runId}/resume`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: clarifyAnswers }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      await readStream(res, handleEvent);
    } catch (e) { setError(String(e)); setPhase('error'); }
  }

  async function sendChat(override?: string) {
    const q = (override ?? chatInput).trim();
    if (!q || chatStreaming) return;
    setChatInput(''); setChatStreaming(true);
    const historySnap = [...chatMessages];
    setChatMessages(prev => [...prev, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    let reply = '';
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: runId, question: q,
          history: historySnap.map(m => ({ role: m.role, content: m.content })) }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await readStream(res, data => {
        if (data.type === 'chunk') {
          reply += data.content as string;
          setChatMessages(prev => [...prev.slice(0, -1), { role: 'assistant', content: reply }]);
        }
      });
    } catch (e) {
      setChatMessages(prev => [...prev.slice(0, -1), { role: 'assistant', content: `Error: ${String(e)}` }]);
    } finally { setChatStreaming(false); }
  }

  async function sendLibraryChat() {
    const q = libraryChatInput.trim();
    if (!q || libraryChatStreaming) return;
    setLibraryChatInput('');
    setLibraryChatStreaming(true);
    setLibraryChunks([]);
    setLibraryRightTab('steps');
    const historySnap = [...libraryChatMessages];
    setLibraryChatMessages(prev => [...prev, { role: 'user', content: q }, { role: 'assistant', content: '' }]);
    let reply = '';
    const addStep = (type: LibraryStepType, label: string, detail?: string) =>
      setLibrarySteps(prev => [...prev, { id: ++libraryStepIdRef.current, type, label, detail, ts: nowTs() }]);
    try {
      const res = await fetch(`${API}/library/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          history: historySnap.map(m => ({ role: m.role, content: m.content })),
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      let finalSources: LibrarySource[] = [];
      await readStream(res, data => {
        if (data.type === 'searching') {
          addStep('searching', 'Thinking', 'Searching indexed reports…');
        } else if (data.type === 'chunks_retrieved') {
          const chunks = (data.chunks as RetrievedChunk[]) ?? [];
          setLibraryChunks(chunks);
          const titles = [...new Set(chunks.map(c => c.title))];
          addStep('chunks_retrieved', `${chunks.length} Chunks Retrieved`,
            titles.length ? `From: ${titles.join(', ')}` : undefined);
        } else if (data.type === 'generating') {
          addStep('generating', 'Generating Answer', 'Writing response grounded in retrieved context…');
        } else if (data.type === 'chunk') {
          reply += data.content as string;
          setLibraryChatMessages(prev => [
            ...prev.slice(0, -1),
            { role: 'assistant', content: reply, sources: finalSources },
          ]);
        } else if (data.type === 'done') {
          finalSources = (data.sources as LibrarySource[]) ?? [];
          addStep('done', 'Complete');
          setLibraryChatMessages(prev => [
            ...prev.slice(0, -1),
            { role: 'assistant', content: reply, sources: finalSources },
          ]);
        } else if (data.type === 'error') {
          addStep('error', 'Error', data.message as string);
        }
      });
    } catch (e) {
      setLibraryChatMessages(prev => [
        ...prev.slice(0, -1),
        { role: 'assistant', content: `Error: ${String(e)}` },
      ]);
    } finally { setLibraryChatStreaming(false); }
  }

  function reset() {
    setPhase('idle'); setQuery(''); setSubtasks([]); setSources([]);
    setReport(''); setShowReport(false); setChatMessages([]);
    setRunId(''); setLog([]); setError(''); setCollapsedLogs(new Set());
    setSupervisorThinking(''); setSupervisorThinkingExpanded(false);
    setSynthesizingActive(false); setResearchEndTime(null); setCopied(false);
    setUsageStats(null);
    setDebateTurns([]); setDebatingActive(false); setDebateExpanded(true); setDebateStreaming(null);
    setDebateRun(false); setJudgingActive(false); setDebateVerdict(null);
    setGapSubtasks([]); setGapPlanningActive(false);
    setPlanCardsExpanded(true); setGapCardsExpanded(true); setDocxBusy(false);
    maxProgress.current = 0;
    stageProgress.current = { plan: { total: 0, done: 0 }, gap: { total: 0, done: 0 } };
    setClarifyQuestions([]); setClarifyOptions([]); setClarifyAnswers([]);
    setActiveId(null);
    setView('research');
    setLibraryChatMessages([]); setLibraryChatInput('');
  }

  // Switch the main view to a past research session from the sidebar
  function selectEntry(entry: HistoryEntry) {
    setView('research');
    if (entry.id === activeId) return;
    if (phase === 'researching' || phase === 'querying' || phase === 'clarifying') return;
    restoreEntry(entry);
    setActiveId(entry.id);
  }

  // Remove a research session from history (and reset the view if it was active).
  // Also deletes the run + its eval reports server-side, so it disappears from
  // the eval dashboard's "Completed Research Runs" table too.
  function deleteEntry(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!window.confirm('Delete this research from history?')) return;
    const entry = history.find(h => h.id === id);
    setHistory(prev => prev.filter(h => h.id !== id));
    if (id === activeId) reset();
    if (entry?.runId) {
      fetch(`${API}/runs/${entry.runId}`, { method: 'DELETE', headers: { 'X-Client-Id': clientId } })
        .catch(() => { /* best-effort: local history is already updated */ });
    }
  }

  async function loadPublicRun(runId: string) {
    if (phase === 'researching' || phase === 'querying' || phase === 'clarifying') return;
    const res = await fetch(`${API}/runs/${runId}`).catch(() => null);
    if (!res?.ok) return;
    const data = await res.json();
    const entry: HistoryEntry = {
      id: runId,
      runId,
      query: data.query ?? '',
      title: data.title ?? data.query ?? '',
      createdAt: data.started_at ? Date.parse(data.started_at) : Date.now(),
      phase: 'done',
      subtasks: (data.plan?.subtasks ?? []).map((q: string) => ({ question: q, status: 'done' as const, findingsCount: 0 })),
      sources: [],
      log: [],
      report: data.report ?? '',
      showReport: true,
      chatMessages: [],
      usageStats: null,
      debateTurns: data.debate_turns ?? [],
      debateRun: (data.debate_turns ?? []).length > 0,
      debateVerdict: data.debate_verdict ?? null,
    };
    restoreEntry(entry);
    setActiveId(runId);
    setView('research');
    setSidebarOpen(false);
  }

  function copyReport() {
    if (!report) return;
    navigator.clipboard.writeText(report).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  // Assemble the full session (title, usage, questions, debate, verdict,
  // report) into one Markdown document — shared by the .md and .docx downloads
  // so both formats always carry identical content. Pure client-side
  // formatting of state we already have — no API or LLM calls.
  function buildSessionMarkdown(): { markdown: string; basename: string } {
    const modelLabel = (id: string) => modelOptions.find(o => o.id === id)?.label ?? id;
    const lines: string[] = [];

    lines.push(`# ${displayQuery}`, '');
    lines.push(`> **Research query:** ${query}`, '');
    lines.push(`*Generated by MindClash on ${new Date().toLocaleString()}*`, '');

    if (usageStats) {
      lines.push('## Run Summary', '');
      lines.push('| Metric | Value |');
      lines.push('| --- | --- |');
      lines.push(`| Lead model | ${modelLabel(usageStats.leadModel)} |`);
      lines.push(`| Subagent model | ${usageStats.subagentModel} |`);
      lines.push(`| Total tokens | ${usageStats.totalTokens.toLocaleString()} (${usageStats.inputTokens.toLocaleString()} in / ${usageStats.outputTokens.toLocaleString()} out) |`);
      lines.push(`| Estimated cost | $${usageStats.costUsd.toFixed(4)} |`);
      lines.push(`| Total time | ${usageStats.elapsedSeconds.toFixed(1)}s |`);
      lines.push('');
    }

    if (subtasks.length > 0) {
      lines.push('## Research Questions', '');
      subtasks.forEach((s, i) => {
        lines.push(`${i + 1}. ${s.question} *(${s.findingsCount} finding${s.findingsCount !== 1 ? 's' : ''})*`);
      });
      lines.push('');
    }

    if (debateTurns.length > 0) {
      lines.push('## Debate', '');
      for (const t of debateTurns) {
        lines.push(`### Round ${t.round} — ${t.agent === 'advocate' ? 'Proposition' : 'Opposition'} (${modelLabel(t.model)})`, '');
        lines.push(t.content, '');
      }
      if (debateVerdict) {
        const label = debateVerdict.winner === 'draw' ? 'Draw' : `${verdictWinnerLabel(debateVerdict.winner)} wins`;
        lines.push(`### Judge's Verdict — ${label}`, '');
        lines.push(`*Judged by ${modelLabel(debateVerdict.model)}*`, '');
        lines.push('', '| Category | Assessment | Winner |', '|---|---|---|');
        for (const row of debateVerdict.rows) {
          lines.push(`| ${row.category} | ${row.assessment} | ${verdictWinnerLabel(row.winner)} |`);
        }
        lines.push('');
      }
    }

    if (gapSubtasks.length > 0) {
      lines.push('## Follow-up Research — Gaps from the Debate', '');
      gapSubtasks.forEach((s, i) => {
        lines.push(`${i + 1}. ${s.question} *(${s.findingsCount} finding${s.findingsCount !== 1 ? 's' : ''})*`);
      });
      lines.push('');
    }

    lines.push('## Final Report', '');
    lines.push(report, '');

    const slug = (title || query).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'research';
    const date = new Date().toISOString().slice(0, 10);
    return { markdown: lines.join('\n'), basename: `${slug}-${date}` };
  }

  function saveBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadSessionMarkdown() {
    if (!report) return;
    const { markdown, basename } = buildSessionMarkdown();
    saveBlob(new Blob([markdown], { type: 'text/markdown;charset=utf-8' }), `${basename}.md`);
  }

  // Same session content as the .md download, packed into a Word document.
  // The converter (and the docx lib behind it) loads on demand via import().
  async function downloadSessionDocx() {
    if (!report || docxBusy) return;
    setDocxBusy(true);
    try {
      const { markdown, basename } = buildSessionMarkdown();
      const { markdownToDocxBlob } = await import('./lib/markdown-docx');
      saveBlob(await markdownToDocxBlob(markdown), `${basename}.docx`);
    } catch (e) {
      // The done view has no error banner — surface the failure directly
      window.alert(`DOCX export failed: ${String(e)}`);
    } finally {
      setDocxBusy(false);
    }
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="flex h-dvh bg-white text-gray-900 overflow-hidden">

      <Sidebar
        history={history}
        activeId={activeId}
        locked={phase === 'researching' || phase === 'querying' || phase === 'clarifying'}
        mobileOpen={sidebarOpen}
        view={view}
        publicRuns={publicRuns}
        onNewResearch={() => { reset(); setSidebarOpen(false); }}
        onShowEvalDashboard={() => { setView('eval'); setSidebarOpen(false); }}
        onShowLibrary={() => { setView('library'); setSidebarOpen(false); }}
        onSelect={entry => { selectEntry(entry); setSidebarOpen(false); }}
        onDelete={deleteEntry}
        onPublicSelect={loadPublicRun}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0">

        {/* Mobile top bar */}
        <div className="lg:hidden flex-shrink-0 h-12 border-b border-gray-200 flex items-center px-4 gap-3">
          <button
            onClick={() => setSidebarOpen(true)}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors -ml-1"
            aria-label="Open menu"
          >
            <MenuIcon />
          </button>
          <h1 className="text-sm font-bold text-gray-900 tracking-tight">MindClash</h1>
        </div>

        {/* ═══════════ EVAL DASHBOARD ═══════════ */}
        {view === 'eval' && <EvalDashboard apiBase={API} clientId={clientId} />}

        {/* ═══════════ RESEARCH LIBRARY (RAG) ═══════════ */}
        {view === 'library' && (
          <div className="flex-1 flex flex-col min-h-0">

            {/* Header */}
            <div className="flex-shrink-0 h-12 border-b border-gray-200 flex items-center gap-3 px-4 sm:px-6">
              <svg className="w-4 h-4 text-indigo-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
              </svg>
              <span className="text-sm font-semibold text-gray-800">Research Library</span>
              <span className="hidden sm:inline text-xs text-gray-400">· Ask questions across all your past research</span>
              <div className="ml-auto flex items-center gap-2">
                {libraryChatMessages.length > 0 && (
                  <button
                    onClick={() => { setLibraryChatMessages([]); try { localStorage.removeItem(LIBRARY_HISTORY_KEY); } catch {} }}
                    className="text-xs text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg px-2.5 py-1.5 transition-colors font-medium"
                  >
                    Clear history
                  </button>
                )}
                <button
                  onClick={() => setShowLibraryActivityMobile(v => !v)}
                  className="lg:hidden flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg px-2.5 py-1.5 transition-colors"
                >
                  {showLibraryActivityMobile ? 'Hide Panel' : 'Show Panel'}
                </button>
              </div>
            </div>

            {/* Split: chat left + retrieval panel right */}
            <div className="flex-1 flex flex-col lg:flex-row min-h-0">

              {/* ── Chat column ── */}
              <div className={`flex-1 min-h-0 flex-col min-w-0 ${showLibraryActivityMobile ? 'hidden lg:flex' : 'flex'}`}>
                {/* Chat messages */}
                <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-3">
                  {libraryChatMessages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
                      <div className="w-14 h-14 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center">
                        <svg className="w-7 h-7 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-gray-800">Ask your research library</p>
                        <p className="text-xs text-gray-400 mt-1 max-w-xs">
                          Questions are answered using relevant passages retrieved from all your completed research reports.
                        </p>
                      </div>
                      <div className="flex flex-col gap-1.5 text-left w-full max-w-sm">
                        {[
                          'What have I researched about AI trends?',
                          'Summarize findings on market analysis topics',
                          'What sources did my research cite most?',
                        ].map(prompt => (
                          <button
                            key={prompt}
                            onClick={() => { setLibraryChatInput(prompt); }}
                            className="text-left text-xs text-gray-600 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700 transition-colors"
                          >
                            {prompt}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    libraryChatMessages.map((m, i) => (
                      <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`text-sm rounded-xl px-4 py-2.5 max-w-[85%] sm:max-w-[75%] ${
                          m.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {m.role === 'assistant' ? (
                            m.content ? (
                              <>
                                <div className="[&_a]:text-indigo-600 [&_a:hover]:underline [&_p]:mb-2 [&_p:last-child]:mb-0
                                  [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:mb-2 [&_ul]:space-y-1
                                  [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:mb-2 [&_ol]:space-y-1
                                  [&_li]:leading-relaxed [&_strong]:font-semibold [&_strong]:text-gray-900
                                  [&_code]:bg-gray-200 [&_code]:px-1 [&_code]:rounded
                                  [&_table]:w-full [&_table]:border-collapse [&_table]:my-2 [&_table]:text-xs
                                  [&_th]:border [&_th]:border-gray-300 [&_th]:bg-gray-50 [&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold [&_th]:text-gray-700
                                  [&_td]:border [&_td]:border-gray-300 [&_td]:px-3 [&_td]:py-1.5 [&_td]:align-top
                                  [&_tr:nth-child(even)_td]:bg-gray-50/60
                                  overflow-x-auto text-xs leading-relaxed">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                                </div>
                                {m.sources && m.sources.length > 0 && (
                                  <div className="mt-2.5 pt-2 border-t border-gray-200 space-y-0.5">
                                    <p className="text-[10px] uppercase tracking-wide text-gray-400 font-medium mb-1">Sources</p>
                                    {m.sources.map((s, j) => (
                                      <p key={j} className="text-[11px] text-gray-500 truncate">
                                        · {s.title || s.query}
                                      </p>
                                    ))}
                                  </div>
                                )}
                              </>
                            ) : (
                              <span className="flex items-center gap-1.5 text-gray-400 text-xs">
                                <Spinner /> Thinking…
                              </span>
                            )
                          ) : m.content}
                        </div>
                      </div>
                    ))
                  )}
                  <div ref={libraryChatEndRef} />
                </div>

                {/* Input */}
                <div className="flex-shrink-0 border-t border-gray-200 px-4 sm:px-6 py-4">
                  <div className="flex gap-2">
                    <input
                      value={libraryChatInput}
                      onChange={e => setLibraryChatInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendLibraryChat()}
                      placeholder="Ask anything across your past research…"
                      disabled={libraryChatStreaming}
                      className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-400 disabled:bg-gray-50"
                    />
                    <button
                      onClick={sendLibraryChat}
                      disabled={!libraryChatInput.trim() || libraryChatStreaming}
                      className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white rounded-xl px-4 py-2.5 text-sm font-semibold transition-colors"
                    >
                      {libraryChatStreaming ? <Spinner /> : <SendIcon />}
                    </button>
                  </div>
                </div>
              </div>

              {/* ── Right panel: Steps + Chunks ── */}
              <div className={`min-h-0 w-full lg:w-[300px] flex-1 lg:flex-none border-t lg:border-t-0 lg:border-l border-gray-200 bg-gray-50 flex-col ${showLibraryActivityMobile ? 'flex' : 'hidden lg:flex'}`}>
                {/* Tab bar */}
                <div className="flex border-b border-gray-200 text-xs font-semibold flex-shrink-0">
                  {(['steps', 'chunks'] as const).map(tab => (
                    <button
                      key={tab}
                      onClick={() => setLibraryRightTab(tab)}
                      className={`flex-1 py-3 flex items-center justify-center gap-1.5 transition-colors ${
                        libraryRightTab === tab
                          ? 'text-indigo-600 border-b-2 border-indigo-500 bg-white'
                          : 'text-gray-400 hover:text-gray-700'
                      }`}
                    >
                      {tab === 'steps' ? (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                          </svg>
                          Steps
                        </>
                      ) : (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                          </svg>
                          {`Chunks${libraryChunks.length ? ` (${libraryChunks.length})` : ''}`}
                        </>
                      )}
                    </button>
                  ))}
                </div>

                {/* Tab content */}
                {libraryRightTab === 'steps' ? (
                  <div className="flex-1 overflow-y-auto">
                    {librarySteps.length === 0 ? (
                      <p className="text-xs text-gray-400 text-center pt-8 px-4">
                        Steps will appear here when you ask a question.
                      </p>
                    ) : (
                      <div className="relative px-3 py-3">
                        <div className="absolute left-[27px] top-6 bottom-6 w-0.5 bg-gradient-to-b from-indigo-300 via-violet-300 to-emerald-300 opacity-50" />
                        {librarySteps.map(step => {
                          const iconColor = step.type === 'done'
                            ? 'bg-emerald-500'
                            : step.type === 'error'
                            ? 'bg-red-500'
                            : step.type === 'searching'
                            ? 'bg-indigo-500'
                            : step.type === 'chunks_retrieved'
                            ? 'bg-violet-500'
                            : 'bg-blue-500';
                          return (
                            <div key={step.id} className="flex gap-3 py-2.5 relative">
                              <div className="flex-shrink-0 z-10">
                                <div className={`w-5 h-5 rounded-full ${iconColor} flex items-center justify-center`}>
                                  <div className="w-2 h-2 rounded-full bg-white" />
                                </div>
                              </div>
                              <div className="flex-1 min-w-0 bg-white border border-gray-200 rounded-xl px-3 py-2.5 shadow-sm">
                                <div className="flex items-start justify-between gap-2 mb-1">
                                  <span className="text-xs font-semibold text-gray-800 leading-tight">{step.label}</span>
                                  <span className="text-[10px] text-gray-400 flex-shrink-0 whitespace-nowrap font-mono">{step.ts}</span>
                                </div>
                                {step.detail && (
                                  <p className="text-[11px] text-gray-500 leading-relaxed">{step.detail}</p>
                                )}
                              </div>
                            </div>
                          );
                        })}
                        <div ref={libraryStepsEndRef} />
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex-1 overflow-y-auto">
                    {libraryChunks.length === 0 ? (
                      <p className="text-xs text-gray-400 text-center pt-8 px-4">
                        Retrieved chunks will appear here after a query.
                      </p>
                    ) : (
                      <div className="px-3 py-3 space-y-2.5">
                        {libraryChunks.map((chunk, i) => (
                          <div key={i} className="bg-white border border-gray-200 rounded-xl px-3 py-2.5 shadow-sm">
                            <div className="flex items-center gap-1.5 mb-1.5">
                              <span className="text-[10px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-100 rounded-full px-1.5 py-0.5">
                                #{i + 1}
                              </span>
                              <span className="text-[11px] font-semibold text-gray-700 truncate">
                                {chunk.title || 'Untitled Report'}
                              </span>
                            </div>
                            <p className="text-[11px] text-gray-500 leading-relaxed line-clamp-4">
                              {chunk.content}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ═══════════ IDLE / QUERYING / CLARIFYING / ERROR ═══════════ */}
        {view === 'research' && (phase === 'idle' || phase === 'querying' || phase === 'clarifying' || phase === 'error') && (
          <div className="flex-1 flex flex-col">

            {phase !== 'clarifying' ? (
              /* ── Home: hero + centered input ── */
              <div className="flex-1 flex flex-col items-center justify-center gap-4 px-4 sm:px-6 text-center">
                <span className="text-5xl mb-1">📝</span>
                <div className="flex flex-col items-center gap-1">
                  <h2 className="text-2xl font-bold text-gray-900">Start Your Research</h2>
                  <p className="text-gray-400 text-sm max-w-md">
                    Ask a question to begin comprehensive AI-powered deep research
                  </p>
                </div>
                {phase === 'error' && (
                  <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2 max-w-lg">
                    {error}
                  </p>
                )}
                <div className="w-full max-w-2xl mt-2">
                  <div className="flex flex-col border border-gray-300 rounded-2xl bg-white transition-colors">
                    <input
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && phase === 'idle' && startResearch()}
                      placeholder="What are the top AI trends shaping 2026?"
                      disabled={phase === 'querying'}
                      className="w-full bg-transparent px-4 pt-4 pb-2 text-sm focus:outline-none disabled:text-gray-600 disabled:cursor-default min-h-[64px]"
                    />
                    <div className="flex items-center justify-end gap-2 px-3 pb-2.5 pt-1">
                      <div ref={debateInfoRef} className="mr-auto flex items-center gap-0.5 relative">
                        <button
                          type="button"
                          onClick={() => setDebateMode(v => !v)}
                          disabled={phase === 'querying'}
                          className={`flex items-center gap-1.5 text-xs font-medium rounded-lg px-2.5 py-1.5 border transition-colors disabled:cursor-default disabled:opacity-50 focus:outline-none ${
                            debateMode
                              ? 'bg-rose-50 border-rose-200 text-rose-600'
                              : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                          }`}
                        >
                          ⚔️   Debate {debateMode ? 'on' : 'off'}
                        </button>
                        <button
                          type="button"
                          onClick={() => setDebateInfoOpen(v => !v)}
                          aria-label="What is debate mode?"
                          className="flex items-center text-gray-400 hover:text-gray-600 focus:outline-none p-1"
                        >
                          <InfoIcon />
                        </button>
                        {debateInfoOpen && (
                          <div className="absolute bottom-full left-0 mb-2 w-64 bg-gray-900 text-white text-xs text-left leading-relaxed rounded-xl shadow-lg px-3 py-2.5 z-10">
                            When enabled, two AI models challenge each other’s answers, identify weaknesses and missing information, then do extra research before producing a final response. This takes longer but often results in a more thorough and accurate answer.
                            <div className="absolute top-full left-4 w-2 h-2 bg-gray-900 rotate-45 -mt-1" />
                          </div>
                        )}
                      </div>
                      {modelOptions.length > 0 && (
                        <ModelPicker
                          options={modelOptions}
                          value={selectedModel}
                          onChange={setSelectedModel}
                          disabled={phase === 'querying'}
                        />
                      )}
                      <button
                        onClick={() => startResearch()}
                        disabled={!query.trim() || phase === 'querying'}
                        className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-5 py-2 text-sm font-semibold transition-colors whitespace-nowrap focus:outline-none"
                      >
                        {phase === 'querying' ? <><Spinner /> Thinking…</> : <><SendIcon /> Research</>}
                      </button>
                    </div>
                    {debateMode && (
                      <div className="flex flex-col gap-1.5 px-4 pb-3 pt-2 border-t border-gray-100">
                        <span className="hidden sm:block text-[11px] text-gray-400 text-left">
                          Two agents from different AI companies debate the findings
                        </span>
                        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                          {/* Label + picker wrap together as one unit */}
                          <span className="flex items-center gap-1 whitespace-nowrap">
                            <span className="text-[11px] text-gray-400 font-medium">Proposition</span>
                            <ModelPicker
                              options={modelOptions}
                              value={advocateModel}
                              onChange={setAdvocateModel}
                              disabled={phase === 'querying'}
                            />
                          </span>
                          <span className="flex items-center gap-1 whitespace-nowrap">
                            <span className="text-[11px] text-gray-400 font-medium">Opposition</span>
                            <ModelPicker
                              options={modelOptions}
                              value={skepticModel}
                              onChange={setSkepticModel}
                              disabled={phase === 'querying'}
                            />
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Suggested research queries — click to fill the input */}
                {phase === 'idle' && (
                  <div className="w-full max-w-2xl mt-5">
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-medium mb-2.5">
                      Try one of these
                    </p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {SUGGESTED_QUERIES.map(s => (
                        <button
                          key={s}
                          onClick={() => setQuery(s)}
                          className="px-4 py-2 rounded-full border border-gray-200 bg-white text-[13px] text-gray-600 hover:border-blue-400 hover:text-blue-600 hover:shadow-sm transition-all"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* ── Clarification questions ── */
              <div className="flex-1 overflow-y-auto flex flex-col justify-center px-4 sm:px-6 py-8">
                <div className="max-w-2xl mx-auto w-full">
                  <div className="border border-gray-200 rounded-2xl shadow-sm bg-white overflow-hidden">

                    {/* Card header */}
                    <div className="flex items-start justify-between px-4 sm:px-6 pt-5 pb-4 border-b border-gray-100">
                      <div>
                        <p className="text-base font-semibold text-gray-900">A few questions to focus the research</p>
                        <p className="text-xs text-gray-400 mt-0.5">Tap an option or type a custom answer</p>
                      </div>
                      <button
                        onClick={() => {
                          // Remove the incomplete history entry so re-submitting doesn't create a duplicate
                          if (activeId) {
                            const entry = history.find(h => h.id === activeId);
                            if (entry?.runId) setPublicRuns(prev => prev.filter(r => r.id !== entry.runId));
                            setHistory(prev => prev.filter(h => h.id !== activeId));
                            setActiveId(null);
                          }
                          setPhase('idle');
                          setClarifyQuestions([]); setClarifyOptions([]); setClarifyAnswers([]);
                        }}
                        className="ml-4 mt-0.5 flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-full text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
                        aria-label="Cancel"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>

                    {/* Card body */}
                    <div className="px-4 sm:px-6 py-5 space-y-7">
                  {clarifyQuestions.map((q, i) => {
                    const chips = clarifyOptions[i] ?? [];
                    const isChipSelected = chips.includes(clarifyAnswers[i] ?? '');
                    const customVal = isChipSelected ? '' : (clarifyAnswers[i] ?? '');
                    return (
                      <div key={i}>
                        <p className="text-sm font-medium text-gray-800 mb-3">{q}</p>
                        {chips.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-3">
                            {chips.map((opt, j) => (
                              <button
                                key={j}
                                onClick={() => { const a = [...clarifyAnswers]; a[i] = opt; setClarifyAnswers(a); }}
                                className={`px-4 py-1.5 rounded-full border text-sm font-medium transition-colors ${
                                  clarifyAnswers[i] === opt
                                    ? 'bg-blue-600 border-blue-600 text-white shadow-sm'
                                    : 'bg-white border-gray-300 text-gray-700 hover:border-blue-400 hover:text-blue-600'
                                }`}
                              >
                                {opt}
                              </button>
                            ))}
                          </div>
                        )}
                        <input
                          value={customVal}
                          onChange={e => { const a = [...clarifyAnswers]; a[i] = e.target.value; setClarifyAnswers(a); }}
                          onKeyDown={e => e.key === 'Enter' && !clarifyAnswers.some(a => !a.trim()) && submitClarification()}
                          placeholder="Other… (type a custom answer)"
                          className="w-full border border-gray-200 rounded-xl px-3.5 py-2.5 text-sm text-gray-700 bg-gray-50 focus:outline-none placeholder:text-gray-400"
                        />
                      </div>
                    );
                  })}

                  <button
                    onClick={submitClarification}
                    disabled={clarifyAnswers.some(a => !a.trim())}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl py-3 text-sm font-semibold transition-colors"
                  >
                    Start Research
                  </button>
                    </div>{/* /card body */}
                  </div>{/* /card */}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════════ RESEARCHING / DONE ═══════════ */}
        {view === 'research' && (phase === 'researching' || phase === 'done') && (
          <div className="flex-1 flex flex-col min-h-0">

            {/* Header */}
            <div className="flex-shrink-0 h-12 border-b border-gray-200 flex items-center justify-between px-4 sm:px-6 gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <span className="text-sm font-semibold text-gray-800 truncate">Research Session</span>
                {phase === 'researching' && (
                  <span className="flex items-center gap-1.5 text-xs text-emerald-600 font-medium flex-shrink-0">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" /> Live
                  </span>
                )}
              </div>
              {/* Mobile toggle for the activity panel (steps/sources) */}
              <button
                onClick={() => setShowActivityMobile(v => !v)}
                className="lg:hidden flex-shrink-0 flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg px-2.5 py-1.5 transition-colors"
              >
                {showActivityMobile ? 'Hide Activity' : 'Show Activity'}
              </button>
            </div>

            {/* Progress bar + milestone trail — starts at 3%, never goes backward,
                hidden once report revealed */}
            {progressPct < 100 && (
              <div className="flex-shrink-0 border-b border-gray-100 px-4 sm:px-6 py-2.5 bg-gray-50">
                <div className="flex items-center justify-between gap-3 text-xs text-gray-500 mb-1.5">
                  <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5 min-w-0">
                    {milestones.map((m, i) => (
                      <span key={m} className="flex items-center gap-1">
                        {i > 0 && <span className="text-gray-300 mx-0.5">›</span>}
                        {i < milestoneIdx && (
                          <svg className="w-3 h-3 text-emerald-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                        {i === milestoneIdx && (
                          <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse flex-shrink-0" />
                        )}
                        <span className={`text-[11px] font-medium whitespace-nowrap ${
                          i < milestoneIdx ? 'text-emerald-600'
                            : i === milestoneIdx ? 'text-blue-600'
                            : 'text-gray-400'
                        }`}>
                          {m}
                        </span>
                      </span>
                    ))}
                  </div>
                  <span className="font-medium text-gray-700 flex-shrink-0">
                    {progressPct <= 3 ? '--' : `${progressPct}%`}
                  </span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-700 ${progressPct <= 3 ? 'animate-pulse' : ''}`}
                    style={{ width: `${Math.max(progressPct, 3)}%` }}
                  />
                </div>
              </div>
            )}

            {/* Split: center + right panel */}
            <div className="flex-1 flex flex-col lg:flex-row min-h-0">

              {/* ── Center ── */}
              <div ref={centerRef} className={`flex-1 min-h-0 flex-col min-w-0 overflow-y-auto ${showActivityMobile ? 'hidden lg:flex' : 'flex'}`}>

                {/* Query heading */}
                <div className="px-4 sm:px-6 lg:px-8 pt-5 sm:pt-7 pb-5 border-b border-gray-100">
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <h2 className="text-xl sm:text-2xl font-bold text-gray-900 leading-snug">{displayQuery}</h2>
                    {report && (
                    <div className="flex-shrink-0 flex items-center gap-2">
                      <button
                        onClick={downloadSessionMarkdown}
                        title="Download the full session (questions, debate, report) as Markdown"
                        className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 border bg-white border-gray-300 text-gray-600 hover:border-gray-400 hover:text-gray-800 transition-colors"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12m0 0l-4-4m4 4l4-4" />
                        </svg>
                        Download .md
                      </button>
                      <button
                        onClick={downloadSessionDocx}
                        disabled={docxBusy}
                        title="Download the full session (questions, debate, report) as a Word document"
                        className="flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 border bg-white border-gray-300 text-gray-600 hover:border-gray-400 hover:text-gray-800 transition-colors disabled:opacity-50 disabled:cursor-default"
                      >
                        {docxBusy ? <Spinner /> : (
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12m0 0l-4-4m4 4l4-4" />
                          </svg>
                        )}
                        Download .docx
                      </button>
                      <button
                        onClick={copyReport}
                        className={`flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 border transition-colors ${
                          copied
                            ? 'bg-emerald-50 border-emerald-300 text-emerald-700'
                            : 'bg-white border-gray-300 text-gray-600 hover:border-gray-400 hover:text-gray-800'
                        }`}
                      >
                        {copied ? (
                          <>
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                            Copied
                          </>
                        ) : (
                          <>
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            Copy Report
                          </>
                        )}
                      </button>
                    </div>
                    )}
                  </div>
                  {phase === 'researching' && (
                    <p className="mt-2 flex items-center gap-1.5 text-sm text-blue-600">
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" /> Researching…
                    </p>
                  )}
                </div>

                {/* Completion banner */}
                {report && (
                  <div className="mx-4 sm:mx-6 lg:mx-8 mt-4 rounded-2xl border border-amber-200 bg-amber-50 px-5 py-3.5 flex items-center gap-2">
                    <span className="text-base">✨</span>
                    <span className="text-sm font-semibold text-gray-800">
                      Research completed! Final report is ready to display.
                    </span>
                  </div>
                )}

                {/* Usage & cost summary */}
                {phase === 'done' && usageStats && (
                  <div className="mx-4 sm:mx-6 lg:mx-8 mt-4 rounded-2xl border border-gray-200 bg-gray-50 px-5 py-4">
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-medium mb-3">
                      Usage Summary
                    </p>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      <div>
                        <p className="text-[11px] text-gray-400 mb-0.5">Models</p>
                        <p className="text-sm font-semibold text-gray-800">
                          {modelOptions.find(o => o.id === usageStats.leadModel)?.label ?? usageStats.leadModel}
                        </p>
                        <p className="text-[11px] text-gray-400 mt-0.5">+ {usageStats.subagentModel}</p>
                      </div>
                      <div>
                        <p className="text-[11px] text-gray-400 mb-0.5">Total Tokens</p>
                        <p className="text-sm font-semibold text-gray-800">{usageStats.totalTokens.toLocaleString()}</p>
                        <p className="text-[11px] text-gray-400 mt-0.5">
                          {usageStats.inputTokens.toLocaleString()} in / {usageStats.outputTokens.toLocaleString()} out
                        </p>
                      </div>
                      <div>
                        <p className="text-[11px] text-gray-400 mb-0.5">Estimated Cost</p>
                        <p className="text-sm font-semibold text-gray-800">${usageStats.costUsd.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-[11px] text-gray-400 mb-0.5">Total Time</p>
                        <p className="text-sm font-semibold text-gray-800">{usageStats.elapsedSeconds.toFixed(1)}s</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Inline step indicators — live-run only, so the container's
                    padding never leaves a phantom gap on finished sessions */}
                {phase === 'researching' && (
                <div className="px-4 sm:px-6 lg:px-8 pt-4 space-y-3">
                  {/* Planning */}
                  {subtasks.length === 0 && !supervisorThinking && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Spinner />
                      <span>🎯 Planning research strategy and identifying key information sources…</span>
                    </div>
                  )}

                  {/* Supervisor thinking block — only while researching is in progress */}
                  {supervisorThinking && (
                    <div className="text-sm text-gray-700">
                      <p>
                        <span className="text-base mr-1.5">🤔</span>
                        <strong>Supervisor Thinking</strong>
                        {': '}
                        {supervisorThinkingExpanded
                          ? supervisorThinking
                          : supervisorThinking.slice(0, 180) + (supervisorThinking.length > 180 ? '…' : '')}
                      </p>
                      {supervisorThinking.length > 180 && (
                        <button
                          onClick={() => setSupervisorThinkingExpanded(v => !v)}
                          className="mt-1.5 text-blue-600 hover:text-blue-800 text-xs flex items-center gap-1 font-medium"
                        >
                          <span className="text-[10px]">{supervisorThinkingExpanded ? '▼' : '▶'}</span>
                          {supervisorThinkingExpanded ? 'Hide full content' : 'Show full content'}
                        </button>
                      )}
                    </div>
                  )}

                  {/* Analyzing */}
                  {subtasks.length > 0 && subtasks.some(s => s.status === 'pending') && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <span>📊</span>
                      <span>Analyzing findings from multiple sources and cross-referencing information…</span>
                    </div>
                  )}

                  {/* Debating — shown while the adversarial debate loop runs */}
                  {debatingActive && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Spinner />
                      <span>⚖️ Agents are debating the findings — proposition vs opposition…</span>
                    </div>
                  )}

                  {/* Gap planning — the lead distills debate objections into follow-up questions */}
                  {gapPlanningActive && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Spinner />
                      <span>🧩 Identifying evidence gaps from the debate…</span>
                    </div>
                  )}

                  {/* Synthesizing — shown when synthesizingActive is set by backend event */}
                  {synthesizingActive && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Spinner />
                      <span>🧠 Synthesizing findings and preparing comprehensive analysis…</span>
                    </div>
                  )}
                </div>
                )}

                {/* Initial question cards — collapsible like the debate panel, so
                    the plan stays re-openable after the debate/report take the stage */}
                {subtasks.length > 0 && (
                  <SubtaskCards
                    heading="Research Plan"
                    icon="🔎"
                    subtasks={subtasks}
                    expanded={planCardsExpanded}
                    onToggle={() => setPlanCardsExpanded(v => !v)}
                  />
                )}

                {/* Debate panel — appears as soon as the debate starts, visible while
                    researching AND after completion */}
                {(debatingActive || debateTurns.length > 0 || debateStreaming) && (
                  <div className="px-4 sm:px-6 lg:px-8 pt-4">
                    <div className="border border-rose-200 rounded-2xl overflow-hidden">
                      <button
                        onClick={() => setDebateExpanded(v => !v)}
                        className="w-full bg-rose-50 px-5 py-3 flex items-center gap-2 text-left focus:outline-none"
                      >
                        <span className="text-base">⚖️</span>
                        <span className="text-sm font-semibold text-gray-800">
                          Debate ({debateTurns.length} turn{debateTurns.length !== 1 ? 's' : ''})
                        </span>
                        <svg
                          className={`w-4 h-4 text-gray-400 ml-auto transition-transform ${debateExpanded ? 'rotate-180' : ''}`}
                          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                        </svg>
                      </button>
                      {debateExpanded && (
                        <div className="px-3 sm:px-4 py-4 space-y-4">
                          {debateTurns.map((t, i) => (
                            <DebateBubble
                              key={i}
                              agent={t.agent}
                              round={t.round}
                              modelLabel={modelOptions.find(o => o.id === t.model)?.label ?? t.model}
                              content={t.content}
                            />
                          ))}
                          {debateStreaming && (
                            <DebateBubble
                              agent={debateStreaming.agent}
                              round={Math.floor(debateTurns.length / 2) + 1}
                              modelLabel={(() => {
                                const id = debateStreaming.agent === 'advocate' ? advocateModel : skepticModel;
                                return modelOptions.find(o => o.id === id)?.label ?? id;
                              })()}
                              content={debateStreaming.content}
                              streaming
                            />
                          )}
                          {debatingActive && !debateStreaming && (() => {
                            // Between turns: whoever spoke last is about to receive a
                            // rebuttal, so show the other side as waiting for their turn.
                            // Proposition speaks first when nothing has happened yet.
                            const activeAgent: DebateTurn['agent'] | null = debateTurns.length > 0
                              ? debateTurns[debateTurns.length - 1].agent
                              : null;
                            const thinkingAgent: DebateTurn['agent'] = activeAgent === 'advocate' ? 'skeptic' : 'advocate';
                            const id = thinkingAgent === 'advocate' ? advocateModel : skepticModel;
                            const modelLabel = modelOptions.find(o => o.id === id)?.label ?? id;
                            return (
                              <DebateBubble
                                agent={thinkingAgent}
                                round={Math.floor(debateTurns.length / 2) + 1}
                                modelLabel={modelLabel}
                                content=""
                                thinking
                              />
                            );
                          })()}
                          <div ref={debateEndRef} />
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Judge's verdict card — appears as soon as judging starts (with a
                    "thinking" placeholder) so the next step shows up instantly when
                    the debate panel collapses, then fills in once the verdict arrives.
                    Auto-collapses once the final report is ready. */}
                {(judgingActive || debateVerdict) && (
                  <div className="px-4 sm:px-6 lg:px-8 pt-4">
                    <div className="border border-amber-200 rounded-2xl overflow-hidden">
                      <button
                        onClick={() => setVerdictExpanded(v => !v)}
                        className="w-full bg-amber-50 px-5 py-3 flex flex-wrap items-center gap-2 text-left focus:outline-none"
                      >
                        <span className="text-base">🏆</span>
                        <span className="text-sm font-semibold text-gray-800">Judge&apos;s Verdict</span>
                        {debateVerdict ? (
                          <span className={`text-[11px] font-semibold rounded-full px-2.5 py-0.5 border ${
                            debateVerdict.winner === 'proposition'
                              ? 'bg-blue-50 border-blue-200 text-blue-700'
                              : debateVerdict.winner === 'opposition'
                                ? 'bg-amber-100 border-amber-300 text-amber-800'
                                : 'bg-gray-100 border-gray-300 text-gray-600'
                          }`}>
                            {debateVerdict.winner === 'draw' ? 'Draw' : `${verdictWinnerLabel(debateVerdict.winner)} wins`}
                          </span>
                        ) : null}
                        <span className="ml-auto flex items-center gap-2 text-[11px] text-gray-400">
                          {debateVerdict && (
                            <>Judged by {modelOptions.find(o => o.id === debateVerdict.model)?.label ?? debateVerdict.model}</>
                          )}
                          <svg
                            className={`w-4 h-4 text-gray-400 transition-transform ${verdictExpanded ? 'rotate-180' : ''}`}
                            fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                          </svg>
                        </span>
                      </button>
                      {verdictExpanded && (
                        debateVerdict ? (
                          <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left border-collapse">
                              <thead>
                                <tr className="border-t border-amber-100 text-[11px] text-gray-400 uppercase tracking-wide">
                                  <th className="px-5 py-2 font-semibold">Category</th>
                                  <th className="px-5 py-2 font-semibold">Judge&apos;s Assessment</th>
                                  <th className="px-5 py-2 font-semibold whitespace-nowrap">Winner</th>
                                </tr>
                              </thead>
                              <tbody>
                                {debateVerdict.rows.map((row, i) => (
                                  <tr key={i} className="border-t border-amber-100">
                                    <td className="px-5 py-2.5 font-medium text-gray-800 whitespace-nowrap">{row.category}</td>
                                    <td className="px-5 py-2.5 text-gray-600">{row.assessment}</td>
                                    <td className="px-5 py-2.5 whitespace-nowrap">
                                      <span className={`text-[11px] font-semibold rounded-full px-2 py-0.5 border ${
                                        row.winner === 'proposition'
                                          ? 'bg-blue-50 border-blue-200 text-blue-700'
                                          : row.winner === 'opposition'
                                            ? 'bg-amber-100 border-amber-300 text-amber-800'
                                            : 'bg-gray-100 border-gray-300 text-gray-600'
                                      }`}>
                                        {verdictWinnerLabel(row.winner)}
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-sm text-gray-400 px-5 py-4 border-t border-amber-100">
                            <Spinner /> The judge is weighing both sides of the debate…
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                {/* Second finding round — gap questions distilled from the debate */}
                {gapSubtasks.length > 0 && (
                  <SubtaskCards
                    heading="Follow-up Research — Gaps from the Debate"
                    icon="🧩"
                    subtasks={gapSubtasks}
                    expanded={gapCardsExpanded}
                    onToggle={() => setGapCardsExpanded(v => !v)}
                  />
                )}

                {/* Report */}
                {showReport && report && (
                  <div className="px-4 sm:px-6 lg:px-8 pt-4 pb-8">
                    <div className="border border-gray-200 rounded-2xl overflow-hidden">
                      <div className="bg-gray-50 px-5 py-3 border-b border-gray-200 flex items-center gap-2">
                        <span className="text-base">📄</span>
                        <span className="text-sm font-semibold text-gray-800">Final Report</span>
                        <span className="ml-auto text-[11px] text-emerald-600 font-medium bg-emerald-50 border border-emerald-200 rounded-full px-2 py-0.5">
                          Complete
                        </span>
                      </div>
                      <div className="p-4 sm:p-6 text-sm text-gray-800 leading-relaxed
                        [&_h1]:text-2xl [&_h1]:font-bold [&_h1]:text-gray-900 [&_h1]:mb-4 [&_h1]:mt-6
                        [&_h2]:text-xl [&_h2]:font-bold [&_h2]:text-gray-900 [&_h2]:mb-3 [&_h2]:mt-8
                        [&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-gray-800 [&_h3]:mb-2 [&_h3]:mt-5
                        [&_p]:mb-4 [&_p]:leading-relaxed [&_p]:text-gray-800
                        [&_ul]:pl-6 [&_ul]:mb-4 [&_ul]:list-disc [&_ol]:pl-6 [&_ol]:mb-4 [&_ol]:list-decimal [&_li]:mb-2 [&_li]:leading-relaxed
                        [&_a]:text-blue-600 [&_a:hover]:underline [&_a]:break-words
                        [&_strong]:text-gray-900 [&_strong]:font-bold
                        [&_blockquote]:border-l-4 [&_blockquote]:border-gray-200 [&_blockquote]:pl-4 [&_blockquote]:text-gray-600 [&_blockquote]:italic [&_blockquote]:my-4
                        [&_code]:bg-gray-100 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:font-mono
                        [&_pre]:bg-gray-100 [&_pre]:rounded-xl [&_pre]:p-4 [&_pre]:overflow-x-auto [&_pre]:text-xs [&_pre]:my-4
                        [&_hr]:border-gray-200 [&_hr]:my-6
                        [&_table]:w-full [&_table]:border-collapse [&_table]:text-sm
                        [&_td]:border [&_td]:border-gray-200 [&_td]:px-3 [&_td]:py-2
                        [&_th]:border [&_th]:border-gray-200 [&_th]:px-3 [&_th]:py-2 [&_th]:bg-gray-50 [&_th]:font-semibold [&_th]:text-left">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                )}

                {/* Follow-up chat */}
                {phase === 'done' && runId && (
                  <div className="px-4 sm:px-6 lg:px-8 pb-8 border-t border-gray-100 space-y-3 mt-auto pt-6">
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-medium">
                      Follow-up Chat
                    </p>
                    {chatMessages.length > 0 && (
                      <div className="space-y-2 max-h-[32rem] overflow-y-auto">
                        {chatMessages.map((m, i) => (
                          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`text-sm rounded-xl px-4 py-2.5 max-w-[85%] sm:max-w-[75%] ${
                              m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'
                            }`}>
                              {m.role === 'assistant' ? (
                                m.content ? (
                                  <div className="[&_a]:text-blue-600 [&_a:hover]:underline [&_p]:mb-2 [&_p:last-child]:mb-0
                                    [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:mb-2 [&_ul]:space-y-1
                                    [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:mb-2 [&_ol]:space-y-1
                                    [&_li]:leading-relaxed [&_strong]:font-semibold [&_strong]:text-gray-900
                                    [&_code]:bg-gray-200 [&_code]:px-1 [&_code]:rounded text-xs leading-relaxed">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                                  </div>
                                ) : (
                                  <span className="flex items-center gap-1.5 text-gray-400 text-xs">
                                    <Spinner /> Thinking…
                                  </span>
                                )
                              ) : m.content}
                            </div>
                          </div>
                        ))}
                        <div ref={chatEndRef} />
                      </div>
                    )}
                    {chatMessages.length === 0 && (
                      <div className="flex flex-wrap gap-2">
                        {[
                          'Summarize the key findings in 3 bullet points',
                          'What is the strongest evidence here?',
                          'What are the main uncertainties or limitations?',
                          'What should I research next?',
                        ].map(q => (
                          <button
                            key={q}
                            onClick={() => sendChat(q)}
                            disabled={chatStreaming}
                            className="text-xs border border-gray-200 rounded-full px-3 py-1.5 text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-colors disabled:opacity-40"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}
                    <div className="flex gap-2">
                      <input
                        value={chatInput}
                        onChange={e => setChatInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendChat()}
                        placeholder="Ask a follow-up question…"
                        disabled={chatStreaming}
                        className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none disabled:bg-gray-50"
                      />
                      <button
                        onClick={() => sendChat()}
                        disabled={!chatInput.trim() || chatStreaming}
                        className="bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white rounded-xl px-4 py-2.5 text-sm font-semibold transition-colors"
                      >
                        {chatStreaming ? <Spinner /> : <SendIcon />}
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* ── Right panel ── */}
              <div className={`min-h-0 w-full lg:w-[300px] flex-1 lg:flex-none border-t lg:border-t-0 lg:border-l border-gray-200 bg-gray-50 flex-col ${showActivityMobile ? 'flex' : 'hidden lg:flex'}`}>
                <div className="flex border-b border-gray-200 text-xs font-semibold">
                  {(['steps', 'sources'] as const).map(tab => (
                    <button
                      key={tab}
                      onClick={() => setRightTab(tab)}
                      disabled={tab === 'sources' && sources.length === 0}
                      className={`flex-1 py-3 flex items-center justify-center gap-1.5 transition-colors ${
                        rightTab === tab
                          ? 'text-blue-600 border-b-2 border-blue-500 bg-white'
                          : 'text-gray-400 hover:text-gray-700'
                      } ${tab === 'sources' && sources.length === 0 ? 'opacity-40 cursor-not-allowed' : ''}`}
                    >
                      {tab === 'steps' ? (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                          </svg>
                          Thinking Steps
                        </>
                      ) : (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                          </svg>
                          {`Sources${sources.length ? ` (${sources.length})` : ''}`}
                        </>
                      )}
                    </button>
                  ))}
                </div>

                {rightTab === 'steps' ? (
                  <div className="flex-1 overflow-y-auto">
                    {log.length === 0 ? (
                      <p className="text-xs text-gray-400 text-center pt-8 px-4">Steps will appear here as research runs.</p>
                    ) : (
                      <div className="relative px-3 py-3">
                        {/* Vertical timeline line */}
                        <div className="absolute left-[27px] top-6 bottom-6 w-0.5 bg-gradient-to-b from-blue-300 via-purple-300 to-emerald-300 opacity-50" />

                        {log.map((entry, i) => {
                          const expanded = !collapsedLogs.has(entry.id);
                          const isLast = i === log.length - 1;
                          // Use server timestamps when available for accurate per-step timing
                          const getTs = (e: LogEntry) => e.serverTs ?? e.createdAt;
                          // Last step: live-tick while researching, then anchor to the
                          // run's end time (or the step's own ts) so it never sticks on "Running…"
                          const running = isLast && researchEndTime === null && phase === 'researching';
                          const endTs = isLast
                            ? (researchEndTime ?? (running ? Math.max(nowTick, getTs(entry)) : getTs(entry)))
                            : getTs(log[i + 1]);
                          const durationMs = endTs - getTs(entry);
                          const durSec = durationMs < 100 ? '< 0.1' : (durationMs / 1000).toFixed(1);
                          const firstTs = log[0] ? getTs(log[0]) : getTs(entry);
                          const totalSec = ((getTs(entry) - firstTs) / 1000).toFixed(1);
                          return (
                            <div key={entry.id} className="flex gap-3 py-2.5 relative">
                              <div className="flex-shrink-0 z-10">
                                <StepIconCircle type={entry.type} />
                              </div>
                              <div className="flex-1 min-w-0 bg-white border border-gray-200 rounded-xl px-3 py-2.5 shadow-sm">
                                {/* Title + timestamp */}
                                <div className="flex items-start justify-between gap-2 mb-1">
                                  <span className="text-xs font-semibold text-gray-800 leading-tight">
                                    {entry.label}
                                  </span>
                                  <span className="text-[10px] text-gray-400 flex-shrink-0 whitespace-nowrap font-mono">{entry.ts}</span>
                                </div>

                                {/* Timing row */}
                                <div className="text-[10px] mb-1.5">
                                  {running ? (
                                    <span className="text-blue-500 flex items-center gap-1 font-mono">
                                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse inline-block" />
                                      Running… {durSec}s
                                    </span>
                                  ) : (
                                    <span className="font-mono text-gray-500">⏱ {durSec}s; total: {totalSec}s</span>
                                  )}
                                </div>

                                {/* Expandable metadata */}
                                {entry.detail && (
                                  <>
                                    {expanded && (
                                      <p className="text-[11px] text-gray-600 leading-relaxed mb-1.5 bg-gray-50 rounded-lg px-2 py-1.5 border border-gray-100">
                                        {entry.detail}
                                      </p>
                                    )}
                                    <button
                                      onClick={() => toggleCollapse(entry.id)}
                                      className="text-[10px] text-blue-500 hover:text-blue-700 flex items-center gap-0.5 transition-colors"
                                    >
                                      <span className="text-[9px]">{expanded ? '▲' : '▶'}</span> {expanded ? 'Hide metadata' : 'View metadata'}
                                    </button>
                                  </>
                                )}
                              </div>
                            </div>
                          );
                        })}
                        <div ref={logEndRef} />
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex-1 overflow-y-auto p-3 space-y-2">
                    {sources.length === 0 ? (
                      <p className="text-xs text-gray-400 text-center pt-6">Sources appear as subtasks complete.</p>
                    ) : sources.map((url, i) => (
                      <a
                        key={i} href={url} target="_blank" rel="noopener noreferrer"
                        className="flex items-start gap-2.5 bg-white border border-gray-200 rounded-lg px-3 py-2.5 hover:border-blue-400 hover:shadow-sm transition-all group"
                      >
                        <img
                          src={`https://www.google.com/s2/favicons?domain=${getDomain(url)}&sz=16`}
                          alt="" width={16} height={16}
                          className="mt-0.5 flex-shrink-0 rounded"
                          onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
                        />
                        <div className="min-w-0">
                          <p className="text-xs font-medium text-gray-700 group-hover:text-blue-600 transition-colors">{getDomain(url)}</p>
                          <p className="text-[10px] text-gray-400 truncate mt-0.5 max-w-[200px]">{url}</p>
                        </div>
                        <svg className="w-3 h-3 text-gray-300 group-hover:text-blue-400 flex-shrink-0 mt-0.5 ml-auto" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                      </a>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
