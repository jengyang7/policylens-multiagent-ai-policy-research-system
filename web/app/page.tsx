'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type Phase = 'idle' | 'querying' | 'clarifying' | 'researching' | 'done' | 'error';
type LogType = 'start' | 'plan' | 'subtask' | 'synthesis' | 'report' | 'complete' | 'clarify' | 'error';

interface SubtaskState { question: string; status: 'pending' | 'done'; findingsCount: number; }
interface ChatMessage  { role: 'user' | 'assistant'; content: string; }
interface LogEntry     { id: number; type: LogType; label: string; detail?: string; ts: string; createdAt: number; serverTs?: number; }

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

function SendIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 2L11 13" /><path d="M22 2L15 22L11 13L2 9L22 2Z" />
    </svg>
  );
}

function StepIconCircle({ type }: { type: LogType }) {
  type Cfg = { bg: string; color: string; d: string };
  const cfg: Record<LogType, Cfg> = {
    start:    { bg: 'bg-blue-100',   color: 'text-blue-600',   d: 'M21 21l-4.35-4.35M17 11A6 6 0 111 11a6 6 0 0116 0z' },
    plan:     { bg: 'bg-purple-100', color: 'text-purple-600', d: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2' },
    subtask:  { bg: 'bg-green-100',  color: 'text-green-600',  d: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
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
// Sidebar (no progress bar)
// ---------------------------------------------------------------------------

function Sidebar() {
  return (
    <aside className="w-60 flex-shrink-0 bg-gray-50 border-r border-gray-200 flex flex-col select-none">
      {/* Brand */}
      <div className="px-5 py-5 border-b border-gray-200 flex items-center justify-between">
        <h1 className="text-[18px] font-extrabold text-gray-900 tracking-tight leading-tight">
          Deep Research Agent
        </h1>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1 text-sm">
        {/* Research — active: white card with border */}
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white border border-gray-200 text-gray-900 font-semibold shadow-sm">
          {/* Sparkles / AI icon */}
          <svg className="w-[18px] h-[18px] text-gray-700 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
          </svg>
          Research
        </button>

        {/* Compare */}
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-500 hover:bg-gray-100 hover:text-gray-800 transition-colors">
          <svg className="w-[18px] h-[18px] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Compare
        </button>

        {/* History */}
        <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-500 hover:bg-gray-100 hover:text-gray-800 transition-colors">
          <svg className="w-[18px] h-[18px] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          History
        </button>
      </nav>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Home() {
  const [phase, setPhase]     = useState<Phase>('idle');
  const [query, setQuery]     = useState('');
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
  const [expandedLogs,  setExpandedLogs] = useState<Set<number>>(new Set());

  const [supervisorThinking,         setSupervisorThinking]         = useState('');
  const [supervisorThinkingExpanded, setSupervisorThinkingExpanded] = useState(false);
  const [synthesizingActive,         setSynthesizingActive]         = useState(false);
  const [researchEndTime,            setResearchEndTime]            = useState<number | null>(null);
  const [copied,                     setCopied]                     = useState(false);

  const logEndRef  = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [log]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [chatMessages]);

  // Restore session from sessionStorage on mount (survives page refresh)
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('dra_session_v1');
      if (!raw) return;
      const s = JSON.parse(raw) as {
        phase: Phase; query: string; runId: string;
        subtasks: SubtaskState[]; sources: string[];
        log: LogEntry[]; report: string; showReport: boolean;
        chatMessages: ChatMessage[];
      };
      setQuery(s.query ?? '');
      setRunId(s.runId ?? '');
      setSubtasks(s.subtasks ?? []);
      setSources(s.sources ?? []);
      setLog(s.log ?? []);
      setReport(s.report ?? '');
      setShowReport(s.showReport ?? false);
      setChatMessages(s.chatMessages ?? []);
      // Transient phases can't be resumed after a refresh — reset them
      const transient: Phase[] = ['researching', 'querying', 'clarifying'];
      const p: Phase = transient.includes(s.phase)
        ? (s.report ? 'done' : 'idle')
        : (s.phase ?? 'idle');
      setPhase(p);
    } catch { /* ignore parse errors */ }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist session to sessionStorage whenever meaningful state changes
  useEffect(() => {
    try {
      sessionStorage.setItem('dra_session_v1', JSON.stringify({
        phase, query, runId, subtasks, sources, log, report, showReport, chatMessages,
      }));
    } catch { /* ignore quota errors */ }
  }, [phase, query, runId, subtasks, sources, log, report, showReport, chatMessages]);

  // Progress — starts at 3 (tiny pulse), never 100 until report revealed, never goes backward
  const progressPct = (() => {
    if (showReport) return 100;
    if (report) return 95;
    if (synthesizingActive) return 92;
    if (subtasks.length === 0) return phase === 'researching' ? 3 : 0; // 3% = tiny visible sliver
    const sub = Math.round(subtasks.filter(s => s.status === 'done').length / subtasks.length * 75);
    return 15 + sub; // 15% when plan arrives → up to 90% — always > 3%, never backward
  })();

  const displayQuery = query ? query.charAt(0).toUpperCase() + query.slice(1) : '';

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function addLog(type: LogType, label: string, detail?: string, serverTs?: number) {
    setLog(prev => [...prev, mkLog(type, label, detail, serverTs)]);
  }

  function toggleExpand(id: number) {
    setExpandedLogs(prev => {
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
      setRunId(data.run_id as string);
      addLog('start', 'Initialization', undefined, serverTs);
    } else if (type === 'plan_thinking') {
      setPhase('researching');
      setSupervisorThinking(data.content as string);
      addLog('plan', 'Research Planning', data.content as string, serverTs);
    } else if (type === 'plan') {
      setPhase(p => (p === 'querying' ? 'researching' : p));
      const qs = (data.subtasks as string[]) ?? [];
      setSubtasks(qs.map(q => ({ question: q, status: 'pending', findingsCount: 0 })));
    } else if (type === 'subtask_done') {
      const q     = data.question as string;
      const count = (data.findings_count as number) ?? 0;
      const srcs  = (data.sources as string[]) ?? [];
      setSubtasks(prev => prev.map(s => s.question === q ? { ...s, status: 'done', findingsCount: count } : s));
      setSources(prev => { const set = new Set(prev); srcs.forEach(s => set.add(s)); return [...set]; });
      addLog('subtask', 'Research Execution', `${count} finding${count !== 1 ? 's' : ''} · ${q.slice(0, 80)}${q.length > 80 ? '…' : ''}`, serverTs);
    } else if (type === 'synthesizing') {
      setSynthesizingActive(true);
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
      addLog('report', 'Final Report', undefined, serverTs);
    } else if (type === 'done') {
      const endTs = serverTs ?? Date.now();
      setPhase('done');
      setResearchEndTime(endTs);
      setSubtasks(prev => prev.map(s => s.status === 'pending' ? { ...s, status: 'done' } : s));
      addLog('complete', 'Completed', undefined, serverTs);
    } else if (type === 'error') {
      setError(data.message as string);
      setPhase('error');
      addLog('error', 'Error', data.message as string, serverTs);
    }
  }

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------

  async function startResearch() {
    if (!query.trim()) return;
    setPhase('querying');
    setSubtasks([]); setSources([]); setReport(''); setShowReport(false);
    setError(''); setChatMessages([]); setRunId('');
    setSupervisorThinking(''); setSupervisorThinkingExpanded(false);
    setSynthesizingActive(false); setResearchEndTime(null);
    setClarifyQuestions([]); setClarifyOptions([]); setClarifyAnswers([]);
    setLog([mkLog('start', 'Initialization', `Query: ${query.trim().slice(0, 120)}`)]);
    setRightTab('steps');
    setExpandedLogs(new Set());
    try {
      const res = await fetch(`${API}/research`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
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

  async function sendChat() {
    const q = chatInput.trim();
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

  function reset() {
    setPhase('idle'); setQuery(''); setSubtasks([]); setSources([]);
    setReport(''); setShowReport(false); setChatMessages([]);
    setRunId(''); setLog([]); setError(''); setExpandedLogs(new Set());
    setSupervisorThinking(''); setSupervisorThinkingExpanded(false);
    setSynthesizingActive(false); setResearchEndTime(null); setCopied(false);
    setClarifyOptions([]);
    try { sessionStorage.removeItem('dra_session_v1'); } catch { /* ignore */ }
  }

  function copyReport() {
    if (!report) return;
    navigator.clipboard.writeText(report).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="flex h-screen bg-white text-gray-900 overflow-hidden">

      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">

        {/* ═══════════ IDLE / QUERYING / CLARIFYING / ERROR ═══════════ */}
        {(phase === 'idle' || phase === 'querying' || phase === 'clarifying' || phase === 'error') && (
          <div className="flex-1 flex flex-col">

            {phase !== 'clarifying' ? (
              /* ── Home: hero + centered input ── */
              <div className="flex-1 flex flex-col items-center justify-center gap-4 px-6 text-center">
                <span className="text-5xl mb-1">🔍</span>
                <h2 className="text-2xl font-bold text-gray-900">Start Your Research</h2>
                <p className="text-gray-400 text-sm max-w-md">
                  Ask a question to begin comprehensive AI-powered research
                </p>
                {phase === 'error' && (
                  <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2 max-w-lg">
                    {error}
                  </p>
                )}
                <div className="flex items-center gap-3 w-full max-w-2xl mt-2">
                  <input
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && phase === 'idle' && startResearch()}
                    placeholder="What are the top AI trends shaping 2026?"
                    disabled={phase === 'querying'}
                    className="flex-1 border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition-colors disabled:bg-gray-50 disabled:text-gray-600 disabled:cursor-default"
                  />
                  <button
                    onClick={startResearch}
                    disabled={!query.trim() || phase === 'querying'}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-5 py-3 text-sm font-semibold transition-colors whitespace-nowrap"
                  >
                    {phase === 'querying' ? <><Spinner /> Thinking…</> : <><SendIcon /> Research</>}
                  </button>
                </div>
              </div>
            ) : (
              /* ── Clarification questions ── */
              <div className="flex-1 overflow-y-auto flex flex-col justify-center px-6 py-8">
                <div className="max-w-2xl mx-auto w-full">
                  <div className="border border-gray-200 rounded-2xl shadow-sm bg-white overflow-hidden">

                    {/* Card header */}
                    <div className="flex items-start justify-between px-6 pt-5 pb-4 border-b border-gray-100">
                      <div>
                        <p className="text-base font-semibold text-gray-900">A few questions to focus the research</p>
                        <p className="text-xs text-gray-400 mt-0.5">Tap an option or type a custom answer</p>
                      </div>
                      <button
                        onClick={() => { setPhase('idle'); setClarifyQuestions([]); setClarifyOptions([]); setClarifyAnswers([]); }}
                        className="ml-4 mt-0.5 flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-full text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors"
                        aria-label="Cancel"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>

                    {/* Card body */}
                    <div className="px-6 py-5 space-y-7">
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
                          className="w-full border border-gray-200 rounded-xl px-3.5 py-2.5 text-sm text-gray-700 bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400 placeholder:text-gray-400"
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
        {(phase === 'researching' || phase === 'done') && (
          <div className="flex-1 flex flex-col min-h-0">

            {/* Header */}
            <div className="flex-shrink-0 h-12 border-b border-gray-200 flex items-center px-6 gap-3">
              <span className="text-sm font-semibold text-gray-800">Research Session</span>
              {phase === 'researching' && (
                <span className="flex items-center gap-1.5 text-xs text-emerald-600 font-medium">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" /> Live
                </span>
              )}
              {phase === 'done' && (
                <button
                  onClick={reset}
                  className="ml-auto flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg px-3 py-1.5 transition-colors"
                >
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
                  </svg>
                  New Research
                </button>
              )}
            </div>

            {/* Progress bar — starts at 3%, never goes backward, 100% only when report revealed */}
            <div className="flex-shrink-0 border-b border-gray-100 px-6 py-2.5 bg-gray-50">
              <div className="flex items-center justify-between text-xs text-gray-500 mb-1.5">
                <span>Progress</span>
                <span className="font-medium text-gray-700">
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

            {/* Split: center + right panel */}
            <div className="flex-1 flex min-h-0">

              {/* ── Center ── */}
              <div className="flex-1 flex flex-col min-w-0 overflow-y-auto">

                {/* Query heading */}
                <div className="px-8 pt-7 pb-5 border-b border-gray-100">
                  <div className="flex items-start justify-between gap-4">
                    <h2 className="text-2xl font-bold text-gray-900 leading-snug">{displayQuery}</h2>
                    {report && (
                      <button
                        onClick={copyReport}
                        className={`flex-shrink-0 flex items-center gap-1.5 text-xs font-semibold rounded-lg px-3 py-1.5 border transition-colors ${
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
                  <div className="mx-8 mt-6 rounded-2xl border border-amber-200 bg-amber-50 px-5 py-3.5 flex items-center gap-2">
                    <span className="text-base">✨</span>
                    <span className="text-sm font-semibold text-gray-800">
                      Research completed! Final report is ready to display.
                    </span>
                  </div>
                )}

                {/* Inline step indicators */}
                <div className="px-8 pt-5 pb-2 space-y-3">
                  {/* Planning */}
                  {phase === 'researching' && subtasks.length === 0 && !supervisorThinking && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Spinner />
                      <span>🎯 Planning research strategy and identifying key information sources…</span>
                    </div>
                  )}

                  {/* Supervisor thinking block */}
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

                  {/* Synthesizing — shown when synthesizingActive is set by backend event */}
                  {synthesizingActive && (
                    <div className="flex items-center gap-2 text-sm text-gray-500">
                      <Spinner />
                      <span>🧠 Synthesizing findings and preparing comprehensive analysis…</span>
                    </div>
                  )}
                </div>

                {/* Subtask cards — hidden once the final report is ready */}
                {subtasks.length > 0 && !(showReport && report) && (
                  <div className="px-8 pt-3 pb-4 space-y-2">
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-medium mb-3">
                      Research Plan
                    </p>
                    {subtasks.map((s, i) => (
                      <div
                        key={i}
                        className={`flex items-start gap-3 px-4 py-3 rounded-xl border transition-colors ${
                          s.status === 'done' ? 'bg-emerald-50 border-emerald-200' : 'bg-gray-50 border-gray-200'
                        }`}
                      >
                        {s.status === 'done'
                          ? <span className="mt-1.5 w-2 h-2 rounded-full flex-shrink-0 bg-emerald-500" />
                          : <span className="mt-0.5 flex-shrink-0"><Spinner /></span>
                        }
                        <span className={`flex-1 text-sm leading-relaxed ${
                          s.status === 'done' ? 'text-emerald-800' : 'text-gray-600'
                        }`}>{s.question.charAt(0).toUpperCase() + s.question.slice(1)}</span>
                        {s.status === 'done' && (
                          <span className="text-[11px] text-emerald-600 font-medium whitespace-nowrap">
                            {s.findingsCount} finding{s.findingsCount !== 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Report */}
                {showReport && report && (
                  <div className="px-8 pt-5 pb-8">
                    <div className="border border-gray-200 rounded-2xl overflow-hidden">
                      <div className="bg-gray-50 px-5 py-3 border-b border-gray-200 flex items-center gap-2">
                        <span className="text-base">📄</span>
                        <span className="text-sm font-semibold text-gray-800">Final Report</span>
                        <span className="ml-auto text-[11px] text-emerald-600 font-medium bg-emerald-50 border border-emerald-200 rounded-full px-2 py-0.5">
                          Complete
                        </span>
                      </div>
                      <div className="p-6 text-sm text-gray-800 leading-relaxed
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
                  <div className="px-8 pb-8 border-t border-gray-100 space-y-3 mt-auto pt-6">
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-medium">
                      Follow-up Chat
                    </p>
                    {chatMessages.length > 0 && (
                      <div className="space-y-2 max-h-60 overflow-y-auto">
                        {chatMessages.map((m, i) => (
                          <div key={i} className={`text-sm rounded-xl px-4 py-2.5 ${
                            m.role === 'user' ? 'bg-blue-600 text-white ml-12' : 'bg-gray-100 text-gray-800 mr-12'
                          }`}>
                            {m.role === 'assistant' ? (
                              m.content ? (
                                <div className="[&_a]:text-blue-600 [&_a:hover]:underline [&_p]:mb-1 [&_code]:bg-gray-200 [&_code]:px-1 [&_code]:rounded text-xs leading-relaxed">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                                </div>
                              ) : (
                                <span className="flex items-center gap-1.5 text-gray-400 text-xs">
                                  <Spinner /> Thinking…
                                </span>
                              )
                            ) : m.content}
                          </div>
                        ))}
                        <div ref={chatEndRef} />
                      </div>
                    )}
                    <div className="flex gap-2">
                      <input
                        value={chatInput}
                        onChange={e => setChatInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendChat()}
                        placeholder="Ask a follow-up question…"
                        disabled={chatStreaming}
                        className="flex-1 border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 disabled:bg-gray-50"
                      />
                      <button
                        onClick={sendChat}
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
              <div className="w-[300px] flex-shrink-0 border-l border-gray-200 bg-gray-50 flex flex-col">
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
                          const expanded = expandedLogs.has(entry.id);
                          const isLast = i === log.length - 1;
                          // Use server timestamps when available for accurate per-step timing
                          const getTs = (e: LogEntry) => e.serverTs ?? e.createdAt;
                          const endTs = isLast ? researchEndTime : getTs(log[i + 1]);
                          const durationMs = endTs !== null && endTs !== undefined ? endTs - getTs(entry) : null;
                          const durSec = durationMs === null
                            ? null
                            : durationMs < 100 ? '< 0.1' : (durationMs / 1000).toFixed(1);
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
                                  {durSec !== null ? (
                                    <span className="font-mono text-gray-500">⏱ {durSec}s; total: {totalSec}s</span>
                                  ) : (
                                    <span className="text-blue-500 flex items-center gap-1">
                                      <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse inline-block" />
                                      Running…
                                    </span>
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
                                      onClick={() => toggleExpand(entry.id)}
                                      className="text-[10px] text-blue-500 hover:text-blue-700 flex items-center gap-0.5 transition-colors"
                                    >
                                      <span className="text-[9px]">{expanded ? '▼' : '▶'}</span> View metadata
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
