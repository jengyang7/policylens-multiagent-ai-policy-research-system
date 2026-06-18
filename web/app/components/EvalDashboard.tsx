'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Types (mirror api/main.py response shapes + eval/schema.py)
// ---------------------------------------------------------------------------

interface RagChunkVerdict {
  chunk_index: number;
  title: string;
  section: string;
  preview: string;
  relevant: boolean;
  reasoning: string;
}

interface RagAnswerClaimVerdict {
  claim: string;
  supported: boolean;
  reasoning: string;
}

interface RagEvalReport {
  question: string;
  generated_at: string;
  selected_reports: string[];
  chunks_retrieved: number;
  chunk_verdicts: RagChunkVerdict[];
  context_precision: number;
  answer: string;
  claim_verdicts: RagAnswerClaimVerdict[];
  answer_faithfulness: number;
  eval_model: string;
  eval_cost_usd: number;
  eval_input_tokens: number;
  eval_output_tokens: number;
}

interface RunStats {
  lead_model?: string;
  subagent_model?: string;
  input_tokens?: number;
  output_tokens?: number;
  cached_tokens?: number;
  total_tokens?: number;
  cost_usd?: number;
  elapsed_seconds?: number;
}

interface RunSummary {
  id: string;
  query: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  stats: RunStats | null;
}

// Counts needed to derive grounding/faithfulness rate — shared by per-run
// reports and the unscoped community trend points.
interface RateCounts {
  total_findings: number;
  ungrounded_count: number;
  total_citations: number;
  unfaithful_count: number;
}

interface EvalReportSummary extends RateCounts {
  id: string;
  run_id: string;
  query: string;
  generated_at: string;
  passed: boolean;
  uncited_count: number;
  failure_reasons: string[];
  eval_model: string;
  eval_cost_usd: number;
  recall_score: number;
  relevance_score: number;
}

// One point on the public "Quality Over Time" trend — aggregated across all
// visitors, with no query text or identifiers (see GET /eval/reports/community).
interface CommunityTrendPoint extends RateCounts {
  generated_at: string;
}

interface GroundingResult {
  subtask: string;
  claim: string;
  evidence_span: string;
  citation_url: string;
  grounded: boolean;
  similarity: number;
  method: string;
  fetch_chars: number;
  note?: string | null;
}

interface FaithfulnessVerdict {
  citation_index: number;
  report_sentence: string;
  matched_finding_claims: string[];
  faithful: boolean;
  confidence: number;
  reasoning: string;
}

interface UncitedSentence {
  sentence: string;
  section: string;
}

interface SubtopicCoverage {
  subtopic: string;
  covered: boolean;
  note: string;
}

interface CompletenessResult {
  subtopics: SubtopicCoverage[];
  recall_score: number;
}

interface RelevanceResult {
  score: number;
  reasoning: string;
}

interface EvalReportDetail extends EvalReportSummary {
  report: {
    grounding_results: GroundingResult[];
    faithfulness_results: FaithfulnessVerdict[];
    uncited_sentences: UncitedSentence[];
    completeness: CompletenessResult;
    relevance: RelevanceResult;
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groundingRate(r: RateCounts): number | null {
  if (r.total_findings === 0) return null;
  return ((r.total_findings - r.ungrounded_count) / r.total_findings) * 100;
}

function faithfulnessRate(r: RateCounts): number | null {
  if (r.total_citations === 0) return null;
  return ((r.total_citations - r.unfaithful_count) / r.total_citations) * 100;
}

function fmtPct(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(0)}%`;
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

function fmtShortDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function fmtCost(v?: number): string {
  return v === undefined ? '—' : `$${v.toFixed(4)}`;
}

function fmtScore(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(1)}/5.0`;
}

// ---------------------------------------------------------------------------
// Small icons (inline, matching page.tsx conventions)
// ---------------------------------------------------------------------------

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-blue-500 flex-shrink-0" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-red-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Trend chart — lightweight inline SVG, no charting dependency
// ---------------------------------------------------------------------------

// Catmull-Rom to cubic-Bezier conversion (tension 1/6) — smooth curve through
// a series of points, matching the look of typical analytics dashboards.
function smoothLinePath(pts: { x: number; y: number }[]): string {
  if (pts.length === 0) return '';
  if (pts.length === 1) return `M ${pts[0].x} ${pts[0].y}`;
  let d = `M ${pts[0].x} ${pts[0].y}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? i : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2 < pts.length ? i + 2 : i + 1];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
  }
  return d;
}

function TrendChart({ points }: { points: CommunityTrendPoint[] }) {
  // Measure the actual rendered width so the viewBox matches it 1:1 — with
  // preserveAspectRatio="none", a viewBox aspect ratio that doesn't match the
  // rendered box stretches circles into ellipses and squashes/elongates text.
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(w);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  if (points.length === 0) {
    return (
      <div ref={containerRef} className="flex items-center justify-center h-48 text-sm text-gray-400">
        No community eval data yet.
      </div>
    );
  }

  const W = width;
  const H = 192; // matches h-48
  const padL = 36;
  const padR = 12;
  const padT = 12;
  const padB = 24;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const n = points.length;
  const xFor = (i: number) => (n === 1 ? padL + innerW / 2 : padL + (i / (n - 1)) * innerW);

  const groundingPts = points.map((p, i) => ({ x: xFor(i), pct: groundingRate(p) }));
  const faithPts = points.map((p, i) => ({ x: xFor(i), pct: faithfulnessRate(p) }));

  // Zoom the y-axis to the data's range (with padding) so small differences
  // near 100% — the common case — are visible instead of a flat line pinned
  // to the top of a fixed 0-100% scale.
  const allPcts = [...groundingPts, ...faithPts]
    .map(p => p.pct)
    .filter((v): v is number => v !== null);
  let yMin = 0;
  let yMax = 100;
  if (allPcts.length > 0) {
    const dataMin = Math.min(...allPcts);
    const dataMax = Math.max(...allPcts);
    const pad = Math.max(dataMax - dataMin, 4) * 0.5;
    yMin = Math.max(0, Math.floor((dataMin - pad) / 5) * 5);
    yMax = Math.min(100, Math.ceil((dataMax + pad) / 5) * 5);
    if (yMin === yMax) {
      yMin = Math.max(0, yMin - 10);
      yMax = Math.min(100, yMax + 10);
    }
  }
  const yFor = (pct: number) => padT + innerH - ((pct - yMin) / (yMax - yMin)) * innerH;

  const linePath = (pts: { x: number; pct: number | null }[]) =>
    smoothLinePath(
      pts.filter((p): p is { x: number; pct: number } => p.pct !== null)
        .map(p => ({ x: p.x, y: yFor(p.pct) })),
    );

  const yTicks = [yMin, (yMin + yMax) / 2, yMax];
  const xTickIdx = n === 1 ? [0] : n === 2 ? [0, 1] : [0, Math.floor((n - 1) / 2), n - 1];

  return (
    <div ref={containerRef}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-48" preserveAspectRatio="none">
        {/* gridlines + y-axis labels, zoomed to the data range */}
        {yTicks.map(pct => (
          <g key={pct}>
            <line x1={padL} y1={yFor(pct)} x2={W - padR} y2={yFor(pct)} stroke="#e5e7eb" strokeWidth={1} />
            <text x={padL - 6} y={yFor(pct) + 3} textAnchor="end" fontSize="9" fill="#9ca3af">{Math.round(pct)}%</text>
          </g>
        ))}

        {/* grounding line (blue, smoothed) */}
        <path d={linePath(groundingPts)} fill="none" stroke="#3b82f6" strokeWidth={2} />
        {/* faithfulness line (amber, smoothed) */}
        <path d={linePath(faithPts)} fill="none" stroke="#f59e0b" strokeWidth={2} />

        {/* donut-style points */}
        {points.map((p, i) => {
          const gPct = groundingRate(p);
          const fPct = faithfulnessRate(p);
          return (
            <g key={i}>
              {gPct !== null && (
                <circle cx={xFor(i)} cy={yFor(gPct)} r={4} fill="#fff" stroke="#3b82f6" strokeWidth={2}>
                  <title>{`Grounding: ${gPct.toFixed(0)}% — ${fmtDate(p.generated_at)}`}</title>
                </circle>
              )}
              {fPct !== null && (
                <circle cx={xFor(i)} cy={yFor(fPct)} r={4} fill="#fff" stroke="#f59e0b" strokeWidth={2}>
                  <title>{`Faithfulness: ${fPct.toFixed(0)}% — ${fmtDate(p.generated_at)}`}</title>
                </circle>
              )}
            </g>
          );
        })}

        {/* x-axis date labels */}
        {xTickIdx.map(i => (
          <text key={i} x={xFor(i)} y={H - 6} textAnchor="middle" fontSize="9" fill="#9ca3af">
            {fmtShortDate(points[i].generated_at)}
          </text>
        ))}
      </svg>
      <div className="flex items-center gap-4 mt-2 text-[11px] text-gray-500">
        <span className="flex items-center gap-1.5">
          <svg width="10" height="10" className="flex-shrink-0"><circle cx="5" cy="5" r="3.5" fill="#fff" stroke="#3b82f6" strokeWidth="2" /></svg>
          Grounding rate
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="10" height="10" className="flex-shrink-0"><circle cx="5" cy="5" r="3.5" fill="#fff" stroke="#f59e0b" strokeWidth="2" /></svg>
          Faithfulness rate
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Detail panel — drill-down for one EvalReportRecord
// ---------------------------------------------------------------------------

function DetailPanel({ detail, loading }: { detail: EvalReportDetail | null; loading: boolean }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-32 text-sm text-gray-400 gap-2">
        <Spinner /> Loading report…
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="flex items-center justify-center h-32 text-sm text-gray-400">
        Click &quot;View&quot; on a run below to see grounding &amp; faithfulness detail.
      </div>
    );
  }

  const { grounding_results, faithfulness_results, uncited_sentences, completeness, relevance } = detail.report;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-gray-900 truncate">{detail.query}</p>
          <p className="text-xs text-gray-400 mt-0.5">{fmtDate(detail.generated_at)}</p>
        </div>
        <span className={`flex-shrink-0 text-xs font-semibold px-2.5 py-1 rounded-full ${
          detail.passed ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-600 border border-red-200'
        }`}>
          {detail.passed ? 'Passed' : 'Failed'}
        </span>
      </div>

      {detail.failure_reasons.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-xl px-3 py-2 text-xs text-red-700 space-y-0.5">
          {detail.failure_reasons.map((reason, i) => <p key={i}>{reason}</p>)}
        </div>
      )}

      {/* Relevance */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Relevance ({relevance.score}/5)
        </p>
        <div className="bg-white border border-gray-200 rounded-lg px-3 py-2">
          <p className="text-xs text-gray-700">{relevance.reasoning}</p>
        </div>
      </div>

      {/* Completeness */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Completeness ({fmtPct(completeness.recall_score * 100)} of subtopics covered)
        </p>
        <div className="space-y-1.5">
          {completeness.subtopics.length === 0 ? (
            <p className="text-xs text-gray-400">No subtopics generated for this query.</p>
          ) : completeness.subtopics.map((s, i) => (
            <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
              {s.covered ? <CheckIcon /> : <CrossIcon />}
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-700">{s.subtopic}</p>
                {s.note && <p className="text-[10px] text-gray-400 mt-0.5">{s.note}</p>}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Grounding results */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Citation Grounding ({grounding_results.length})
        </p>
        <div className="space-y-1.5">
          {grounding_results.length === 0 ? (
            <p className="text-xs text-gray-400">No findings to check.</p>
          ) : grounding_results.map((g, i) => (
            <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
              {g.grounded ? <CheckIcon /> : <CrossIcon />}
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-700 line-clamp-2">{g.evidence_span}</p>
                <p className="text-[10px] text-gray-400 mt-0.5 truncate">
                  {g.method} · sim {g.similarity.toFixed(2)} · {g.citation_url}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Faithfulness verdicts */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Faithfulness Verdicts ({faithfulness_results.length})
        </p>
        <div className="space-y-1.5">
          {faithfulness_results.length === 0 ? (
            <p className="text-xs text-gray-400">No cited sentences to check.</p>
          ) : faithfulness_results.map((f, i) => (
            <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
              {f.faithful ? <CheckIcon /> : <CrossIcon />}
              <div className="min-w-0 flex-1">
                <p className="text-xs text-gray-700 line-clamp-2">[{f.citation_index}] {f.report_sentence}</p>
                <p className="text-[10px] text-gray-400 mt-0.5">{f.reasoning}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Uncited sentences */}
      <div>
        <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
          Uncited Sentences ({uncited_sentences.length})
        </p>
        {uncited_sentences.length === 0 ? (
          <p className="text-xs text-gray-400">Every sentence in the report carries a citation.</p>
        ) : (
          <div className="space-y-1.5">
            {uncited_sentences.map((u, i) => (
              <div key={i} className="bg-white border border-gray-200 rounded-lg px-3 py-2">
                <p className="text-xs text-gray-700 line-clamp-2">{u.sentence}</p>
                <p className="text-[10px] text-gray-400 mt-0.5">{u.section}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// RAG Eval result panel
// ---------------------------------------------------------------------------

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="w-full bg-gray-100 rounded-full h-1.5 mt-1.5">
      <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

function RagEvalResultPanel({ result }: { result: RagEvalReport }) {
  const [showAnswer, setShowAnswer] = useState(false);
  const precisionColor = result.context_precision >= 0.8 ? 'bg-emerald-500' : result.context_precision >= 0.5 ? 'bg-amber-400' : 'bg-red-400';
  const faithColor = result.answer_faithfulness >= 0.8 ? 'bg-emerald-500' : result.answer_faithfulness >= 0.5 ? 'bg-amber-400' : 'bg-red-400';

  return (
    <div className="space-y-4 pt-2 border-t border-gray-100">
      {/* Score cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
          <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Context Precision</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{fmtPct(result.context_precision * 100)}</p>
          <ScoreBar value={result.context_precision} color={precisionColor} />
          <p className="text-[10px] text-gray-400 mt-1.5">
            {result.chunk_verdicts.filter(v => v.relevant).length}/{result.chunk_verdicts.length} chunks relevant
          </p>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-xl px-4 py-3">
          <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Answer Faithfulness</p>
          <p className="text-2xl font-bold text-gray-900 mt-1">{fmtPct(result.answer_faithfulness * 100)}</p>
          <ScoreBar value={result.answer_faithfulness} color={faithColor} />
          <p className="text-[10px] text-gray-400 mt-1.5">
            {result.claim_verdicts.filter(v => v.supported).length}/{result.claim_verdicts.length} claims supported
          </p>
        </div>
      </div>

      {/* Meta row */}
      <div className="flex items-center gap-3 text-[11px] text-gray-400 flex-wrap">
        <span>{result.selected_reports.length} report(s) selected · {result.chunks_retrieved} chunk(s) retrieved</span>
        <span>·</span>
        <span>{fmtCost(result.eval_cost_usd)} eval cost</span>
        <span>·</span>
        <span>{result.eval_model}</span>
      </div>

      {/* Chunk verdicts */}
      {result.chunk_verdicts.length > 0 && (
        <div>
          <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
            Retrieved Chunks
          </p>
          <div className="space-y-1.5">
            {result.chunk_verdicts.map((v, i) => (
              <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
                {v.relevant ? <CheckIcon /> : <CrossIcon />}
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-gray-700">
                    {v.title}{v.section && v.section !== '/' ? <span className="text-gray-400 font-normal"> · {v.section}</span> : null}
                  </p>
                  <p className="text-[11px] text-gray-500 mt-0.5 line-clamp-2">{v.preview}</p>
                  {!v.relevant && <p className="text-[10px] text-red-500 mt-0.5">{v.reasoning}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Claim verdicts */}
      {result.claim_verdicts.length > 0 && (
        <div>
          <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
            Answer Claims
          </p>
          <div className="space-y-1.5">
            {result.claim_verdicts.map((v, i) => (
              <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
                {v.supported ? <CheckIcon /> : <CrossIcon />}
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-gray-700">{v.claim}</p>
                  {!v.supported && <p className="text-[10px] text-red-500 mt-0.5">{v.reasoning}</p>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generated answer (collapsible) */}
      {result.answer && (
        <div>
          <button
            onClick={() => setShowAnswer(a => !a)}
            className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold hover:text-gray-600 transition-colors"
          >
            {showAnswer ? '▾' : '▸'} Generated Answer
          </button>
          {showAnswer && (
            <div className="mt-1.5 bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
              <p className="text-xs text-gray-700 whitespace-pre-wrap">{result.answer}</p>
            </div>
          )}
        </div>
      )}

      {result.chunks_retrieved === 0 && (
        <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          No chunks were retrieved. Try re-indexing the library or asking a question covered by your research reports.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main dashboard
// ---------------------------------------------------------------------------

export default function EvalDashboard({ apiBase, clientId }: { apiBase: string; clientId: string }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [reports, setReports] = useState<EvalReportSummary[]>([]);
  const [communityTrend, setCommunityTrend] = useState<CommunityTrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [runningEvalFor, setRunningEvalFor] = useState<string | null>(null);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EvalReportDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);

  // RAG eval state
  const [ragQuestion, setRagQuestion] = useState('');
  const [ragEvalLoading, setRagEvalLoading] = useState(false);
  const [ragEvalResult, setRagEvalResult] = useState<RagEvalReport | null>(null);

  // Re-index state
  const [reindexing, setReindexing] = useState(false);
  const [reindexResult, setReindexResult] = useState<{ indexed: number; total: number; failed: {run_id: string; reason: string}[] } | null>(null);

  // Custom confirmation modal (replaces window.confirm for destructive actions)
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string;
    message: string;
    confirmLabel: string;
    onConfirm: () => void;
  } | null>(null);

  // Model used to judge faithfulness/completeness/relevance for "Run Eval".
  // Fixed (not user-selectable) so every eval — including ones aggregated into
  // "Community Average" and the trend chart below — is judged consistently.
  // Defaults to a different provider than the report's writer model (see
  // engine/models.py EVAL_MODEL) so the judge isn't grading its own work.
  const [evalModel, setEvalModel] = useState('claude-haiku-4-5');

  useEffect(() => {
    fetch(`${apiBase}/models`)
      .then(res => res.json())
      .then((data: { defaults?: { eval?: string } }) => {
        if (data.defaults?.eval) setEvalModel(data.defaults.eval);
      })
      .catch(() => { /* keep the hardcoded fallback */ });
  }, [apiBase]);

  const headers = useMemo(() => ({ 'X-Client-Id': clientId }), [clientId]);

  const load = useCallback(async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const [runsRes, reportsRes, trendRes] = await Promise.all([
        fetch(`${apiBase}/runs?status=done`, { headers }),
        fetch(`${apiBase}/eval/reports`, { headers }),
        fetch(`${apiBase}/eval/reports/community`),
      ]);
      if (!runsRes.ok || !reportsRes.ok || !trendRes.ok) throw new Error('Failed to load eval data');
      setRuns(await runsRes.json());
      setReports(await reportsRes.json());
      setCommunityTrend(await trendRes.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase, headers]);

  useEffect(() => { load(); }, [load]);

  // Most recent eval report per run_id, for the runs table's status badge.
  const latestByRun = useMemo(() => {
    const map = new Map<string, EvalReportSummary>();
    for (const r of reports) {
      const existing = map.get(r.run_id);
      if (!existing || r.generated_at > existing.generated_at) map.set(r.run_id, r);
    }
    return map;
  }, [reports]);

  const summary = useMemo(() => {
    if (reports.length === 0) return null;
    const passed = reports.filter(r => r.passed).length;
    const totalFindings = reports.reduce((s, r) => s + r.total_findings, 0);
    const ungrounded = reports.reduce((s, r) => s + r.ungrounded_count, 0);
    const totalCitations = reports.reduce((s, r) => s + r.total_citations, 0);
    const unfaithful = reports.reduce((s, r) => s + r.unfaithful_count, 0);
    const totalRecall = reports.reduce((s, r) => s + r.recall_score, 0);
    const totalRelevance = reports.reduce((s, r) => s + r.relevance_score, 0);
    const totalCost = reports.reduce((s, r) => s + r.eval_cost_usd, 0);
    return {
      runsEvaluated: reports.length,
      passRate: (passed / reports.length) * 100,
      groundingRate: totalFindings === 0 ? null : ((totalFindings - ungrounded) / totalFindings) * 100,
      faithfulnessRate: totalCitations === 0 ? null : ((totalCitations - unfaithful) / totalCitations) * 100,
      completenessRate: (totalRecall / reports.length) * 100,
      relevanceScore: totalRelevance / reports.length,
      totalCost,
    };
  }, [reports]);

  const loadDetail = useCallback(async (reportId: string) => {
    setSelectedReportId(reportId);
    setDetailLoading(true);
    try {
      const res = await fetch(`${apiBase}/eval/reports/${reportId}`, { headers });
      if (!res.ok) throw new Error('Failed to load report detail');
      setDetail(await res.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setDetailLoading(false);
    }
  }, [apiBase, headers]);

  async function runEval(runId: string) {
    setRunningEvalFor(runId);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/runs/${runId}/eval?model=${encodeURIComponent(evalModel)}`, {
        method: 'POST',
        headers,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `HTTP ${res.status}`);
      }
      const record: EvalReportDetail = await res.json();
      setDetail(record);
      setSelectedReportId(record.id);
      await load();
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setRunningEvalFor(null);
    }
  }

  async function deleteRun(runId: string) {
    setDeletingRunId(runId);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/runs/${runId}`, { method: 'DELETE', headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (selectedReportId && latestByRun.get(runId)?.id === selectedReportId) {
        setSelectedReportId(null);
        setDetail(null);
      }
      await load();
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setDeletingRunId(null);
    }
  }

  function confirmDeleteRun(runId: string) {
    setConfirmDialog({
      title: 'Delete this research run?',
      message: 'This also removes its eval history. This cannot be undone.',
      confirmLabel: 'Delete',
      onConfirm: () => deleteRun(runId),
    });
  }

  async function runRagEval() {
    if (!ragQuestion.trim()) return;
    setRagEvalLoading(true);
    setRagEvalResult(null);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/library/eval`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: ragQuestion, eval_model: evalModel }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? `HTTP ${res.status}`);
      }
      setRagEvalResult(await res.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setRagEvalLoading(false);
    }
  }

  async function reindexLibrary() {
    setReindexing(true);
    setReindexResult(null);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/library/reindex`, { method: 'POST', headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setReindexResult(await res.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setReindexing(false);
    }
  }

  return (
    <div className="flex-1 overflow-y-auto">
    <div className="p-4 sm:p-6 space-y-6 max-w-5xl mx-auto w-full">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Eval Dashboard</h2>
          <p className="text-sm text-gray-400 mt-0.5">
            Citation grounding &amp; faithfulness checks across completed research runs.
          </p>
        </div>
        <div />
      </div>

      {errorMsg && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-2">
          {errorMsg}
        </p>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-32 text-sm text-gray-400 gap-2">
          <Spinner /> Loading…
        </div>
      ) : (
        <>
          {/* Stats */}
          <div>
            <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-2">Stats</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Runs Evaluated</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary?.runsEvaluated ?? 0}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Pass Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.passRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Grounding Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.groundingRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Faithfulness Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.faithfulnessRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Completeness</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtPct(summary.completenessRate) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Relevance</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtScore(summary.relevanceScore) : '—'}</p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl shadow-sm px-4 py-3">
                <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">Eval Cost</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{summary ? fmtCost(summary.totalCost) : '—'}</p>
              </div>
            </div>
          </div>

          {/* Trend chart */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900 mb-2">
              Quality Over Time
              <span className="normal-case text-gray-400 font-normal text-xs"> — across all visitors</span>
            </p>
            <TrendChart points={communityTrend} />
          </div>

          {/* Runs table */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
            <p className="text-sm font-semibold text-gray-900 px-4 pt-4 pb-2">Recent Research</p>
            {runs.length === 0 ? (
              <p className="text-sm text-gray-400 px-4 pb-4">No completed research runs yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold border-b border-gray-200">
                      <th className="text-left px-4 py-2 font-semibold">Query</th>
                      <th className="text-left px-4 py-2 font-semibold">Started</th>
                      <th className="text-right px-4 py-2 font-semibold">Cost</th>
                      <th className="text-right px-4 py-2 font-semibold">Tokens</th>
                      <th className="text-right px-4 py-2 font-semibold">Elapsed</th>
                      <th className="text-left px-4 py-2 font-semibold">Eval</th>
                      <th className="px-4 py-2"></th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map(run => {
                      const latest = latestByRun.get(run.id);
                      const isRunning = runningEvalFor === run.id;
                      return (
                        <tr key={run.id} className="border-b border-gray-100 last:border-0 hover:bg-gray-50">
                          <td className="px-4 py-2.5 max-w-[260px]">
                            <p className="truncate text-gray-800">{run.query || 'Untitled research'}</p>
                          </td>
                          <td className="px-4 py-2.5 text-gray-500 whitespace-nowrap">{fmtDate(run.started_at)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{fmtCost(run.stats?.cost_usd)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{run.stats?.total_tokens?.toLocaleString() ?? '—'}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{run.stats?.elapsed_seconds !== undefined ? `${run.stats.elapsed_seconds}s` : '—'}</td>
                          <td className="px-4 py-2.5 whitespace-nowrap">
                            {latest ? (
                              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                                latest.passed ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-red-50 text-red-600 border border-red-200'
                              }`}>
                                {latest.passed ? 'Passed' : 'Failed'}
                              </span>
                            ) : (
                              <span className="text-xs text-gray-400">Not evaluated</span>
                            )}
                          </td>
                          <td className="px-4 py-2.5 text-right whitespace-nowrap">
                            <div className="flex items-center justify-end gap-3">
                              {latest ? (
                                <button
                                  onClick={() => loadDetail(latest.id)}
                                  className="text-xs font-semibold text-blue-600 hover:text-blue-800 transition-colors"
                                >
                                  View
                                </button>
                              ) : (
                                <button
                                  onClick={() => runEval(run.id)}
                                  disabled={isRunning}
                                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-800 disabled:opacity-50 transition-colors"
                                >
                                  {isRunning && <Spinner />} {isRunning ? 'Running…' : 'Run Eval'}
                                </button>
                              )}
                              <button
                                onClick={() => confirmDeleteRun(run.id)}
                                disabled={deletingRunId === run.id}
                                title="Delete this research run"
                                className="text-xs font-semibold text-gray-400 hover:text-red-600 disabled:opacity-50 transition-colors"
                              >
                                {deletingRunId === run.id ? '…' : 'Delete'}
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Detail panel */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900 mb-3">Report Detail</p>
            <DetailPanel detail={detail} loading={detailLoading} />
          </div>

          {/* RAG Eval */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4 space-y-4">
            <div className="flex items-start justify-between gap-3 flex-wrap">
              <div>
                <p className="text-sm font-semibold text-gray-900">Research Library RAG Eval</p>
                <p className="text-xs text-gray-400 mt-0.5">
                  Evaluates retrieval precision and answer faithfulness for any question.
                </p>
              </div>
              <div className="flex flex-col items-end gap-1">
                <button
                  onClick={reindexLibrary}
                  disabled={reindexing}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-gray-600 hover:text-gray-900 border border-gray-200 hover:border-gray-300 px-3 py-1.5 rounded-lg disabled:opacity-50 transition-colors"
                >
                  {reindexing && <Spinner />}
                  {reindexing ? 'Re-indexing…' : 'Re-index Library'}
                </button>
                {reindexResult && (
                  <p className="text-[11px] text-gray-500">
                    Indexed {reindexResult.indexed}/{reindexResult.total} runs
                    {reindexResult.failed.length > 0 && ` · ${reindexResult.failed.length} failed`}
                  </p>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <input
                value={ragQuestion}
                onChange={e => setRagQuestion(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && !ragEvalLoading && runRagEval()}
                placeholder="Ask a question to evaluate retrieval quality…"
                className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 transition"
              />
              <button
                onClick={runRagEval}
                disabled={ragEvalLoading || !ragQuestion.trim()}
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg disabled:opacity-50 transition-colors whitespace-nowrap"
              >
                {ragEvalLoading && <Spinner />}
                {ragEvalLoading ? 'Evaluating…' : 'Run RAG Eval'}
              </button>
            </div>

            {ragEvalResult && <RagEvalResultPanel result={ragEvalResult} />}
          </div>
        </>
      )}

      {/* Custom confirmation modal — replaces window.confirm for destructive actions */}
      {confirmDialog && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
          onClick={() => setConfirmDialog(null)}
        >
          <div
            className="bg-white rounded-2xl shadow-xl max-w-sm w-full p-5"
            onClick={e => e.stopPropagation()}
          >
            <p className="text-base font-semibold text-gray-900">{confirmDialog.title}</p>
            <p className="text-sm text-gray-500 mt-1.5">{confirmDialog.message}</p>
            <div className="flex justify-end gap-2 mt-5">
              <button
                onClick={() => setConfirmDialog(null)}
                className="text-sm font-medium text-gray-600 hover:text-gray-800 px-3 py-1.5 rounded-lg hover:bg-gray-100 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => { confirmDialog.onConfirm(); setConfirmDialog(null); }}
                className="text-sm font-semibold text-white bg-red-600 hover:bg-red-700 px-3.5 py-1.5 rounded-lg transition-colors"
              >
                {confirmDialog.confirmLabel}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
    </div>
  );
}
