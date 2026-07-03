'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Types (mirror api/main.py response shapes + eval/schema.py)
// ---------------------------------------------------------------------------

interface RunStats {
  lead_model?: string;
  subagent_model?: string;
  mode?: string;
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
  title?: string | null;
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
  authority_score: number;
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

interface CitationCoverageIssue {
  sentence: string;
  section: string;
  reasoning: string;
}

interface CitationCoverageResult {
  coverage_score: number;
  cited_sentence_count: number;
  uncited_factual_claims: CitationCoverageIssue[];
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

interface SourceAuthorityVerdict {
  url: string;
  domain: string;
  tier: 'primary' | 'secondary' | 'other';
  reasoning: string;
}

interface SourceAuthorityResult {
  verdicts: SourceAuthorityVerdict[];
  authority_score: number;
  primary_count: number;
  secondary_count: number;
  other_count: number;
}

interface EvalReportDetail extends EvalReportSummary {
  report: {
    grounding_results: GroundingResult[];
    faithfulness_results: FaithfulnessVerdict[];
    uncited_sentences: UncitedSentence[];
    citation_coverage?: CitationCoverageResult;
    completeness: CompletenessResult;
    relevance: RelevanceResult;
    source_authority?: SourceAuthorityResult; // absent on eval reports predating the metric
  };
}

interface ModelOption {
  id: string;
  label: string;
  description: string;
}

interface ModelsResponse {
  default: string;
  defaults?: { lead?: string; eval?: string; advocate?: string; skeptic?: string };
  options: ModelOption[];
}

interface BenchmarkRunProgress {
  stage: string;
  detail?: string;
  percent: number;
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

function fmtCost(v?: number): string {
  return v === undefined ? '—' : `$${v.toFixed(4)}`;
}

function fmtScore(v: number | null): string {
  return v === null ? '—' : `${v.toFixed(1)}/5.0`;
}

function modeLabel(mode?: string): string {
  const labels: Record<string, string> = {
    single_agent: 'Legacy Run',
    multi_agent_no_compaction: 'Legacy Run',
    multi_agent_compaction: 'Legacy Run',
    multi_agent_verified: 'Standard Research',
    debate_gap: 'Debate Research',
    legacy: 'Legacy Run',
    unknown: 'Legacy Run',
  };
  return labels[mode ?? ''] ?? 'Unknown';
}

function modeRank(mode?: string): number {
  const ranks: Record<string, number> = {
    multi_agent_verified: 1,
    debate_gap: 2,
  };
  return ranks[mode ?? ''] ?? 99;
}

function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 72) || 'eval-report';
}

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

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
      if (!line.startsWith('data: ')) continue;
      try { onEvent(JSON.parse(line.slice(6))); } catch { /* skip malformed SSE line */ }
    }
  }
}

function buildEvalDetailMarkdown(detail: EvalReportDetail): string {
  const grounding = groundingRate(detail);
  const faithfulness = faithfulnessRate(detail);
  const lines: string[] = [];
  const report = detail.report;

  lines.push(`# Eval Report: ${detail.query}`, '');
  lines.push(`- Run ID: ${detail.run_id}`);
  lines.push(`- Eval report ID: ${detail.id}`);
  lines.push(`- Generated: ${new Date(detail.generated_at).toLocaleString()}`);
  lines.push(`- Result: ${detail.passed ? 'Passed' : 'Failed'}`);
  lines.push(`- Eval model: ${detail.eval_model}`);
  lines.push(`- Eval cost: ${fmtCost(detail.eval_cost_usd)}`);
  lines.push('');

  lines.push('## Metrics', '');
  lines.push('| Metric | Value |');
  lines.push('| --- | ---: |');
  lines.push(`| Citation grounding | ${fmtPct(grounding)} |`);
  lines.push(`| Citation faithfulness | ${fmtPct(faithfulness)} |`);
  lines.push(`| Completeness / recall | ${fmtPct(detail.recall_score * 100)} |`);
  lines.push(`| Relevance | ${fmtScore(detail.relevance_score)} |`);
  if (detail.report.source_authority) {
    lines.push(`| Source authority | ${fmtPct(detail.report.source_authority.authority_score * 100)} |`);
  }
  lines.push(`| Unsupported findings | ${detail.ungrounded_count}/${detail.total_findings} |`);
  lines.push(`| Unfaithful citations | ${detail.unfaithful_count}/${detail.total_citations} |`);
  lines.push(`| Uncited sentences | ${detail.uncited_count} |`);
  lines.push('');

  if (detail.failure_reasons.length > 0) {
    lines.push('## Failure Reasons', '');
    detail.failure_reasons.forEach(reason => lines.push(`- ${reason}`));
    lines.push('');
  }

  lines.push('## Relevance', '');
  lines.push(`Score: ${report.relevance.score}/5`, '');
  lines.push(report.relevance.reasoning, '');

  lines.push('## Completeness', '');
  lines.push(`Recall: ${fmtPct(report.completeness.recall_score * 100)}`, '');
  if (report.completeness.subtopics.length === 0) {
    lines.push('No subtopics generated.', '');
  } else {
    report.completeness.subtopics.forEach((s, i) => {
      lines.push(`${i + 1}. ${s.covered ? '[covered]' : '[missing]'} ${s.subtopic}`);
      if (s.note) lines.push(`   - ${s.note}`);
    });
    lines.push('');
  }

  lines.push('## Citation Grounding', '');
  if (report.grounding_results.length === 0) {
    lines.push('No findings to check.', '');
  } else {
    report.grounding_results.forEach((g, i) => {
      lines.push(`### Finding ${i + 1}: ${g.grounded ? 'Grounded' : 'Ungrounded'}`);
      lines.push(`- Source: ${g.citation_url}`);
      lines.push(`- Method: ${g.method}`);
      lines.push(`- Similarity: ${g.similarity.toFixed(2)}`);
      if (g.note) lines.push(`- Note: ${g.note}`);
      lines.push('');
      lines.push('Evidence span:');
      lines.push('```');
      lines.push(g.evidence_span);
      lines.push('```', '');
    });
  }

  lines.push('## Faithfulness Verdicts', '');
  if (report.faithfulness_results.length === 0) {
    lines.push('No cited sentences to check.', '');
  } else {
    report.faithfulness_results.forEach((f, i) => {
      lines.push(`### Citation ${i + 1}: ${f.faithful ? 'Faithful' : 'Unfaithful'}`);
      lines.push(`- Citation index: [${f.citation_index}]`);
      lines.push(`- Confidence: ${f.confidence.toFixed(2)}`);
      lines.push(`- Reasoning: ${f.reasoning}`);
      if (f.matched_finding_claims.length > 0) {
        lines.push('- Matched finding claims:');
        f.matched_finding_claims.forEach(claim => lines.push(`  - ${claim}`));
      }
      lines.push('');
      lines.push('Report sentence:');
      lines.push('```');
      lines.push(f.report_sentence);
      lines.push('```', '');
    });
  }

  if (report.citation_coverage) {
    lines.push('## Citation Coverage', '');
    lines.push(`Coverage score: ${fmtPct(report.citation_coverage.coverage_score * 100)}`, '');
    if (report.citation_coverage.uncited_factual_claims.length === 0) {
      lines.push('No uncited factual claims were flagged.', '');
    } else {
      report.citation_coverage.uncited_factual_claims.forEach((issue, i) => {
        lines.push(`${i + 1}. ${issue.sentence}`);
        lines.push(`   - Section: ${issue.section}`);
        lines.push(`   - Reasoning: ${issue.reasoning}`);
      });
      lines.push('');
    }
  }

  lines.push('## Uncited Sentences', '');
  if (report.uncited_sentences.length === 0) {
    lines.push('Every sentence in the report carries a citation.', '');
  } else {
    report.uncited_sentences.forEach((u, i) => {
      lines.push(`${i + 1}. ${u.sentence}`);
      lines.push(`   - Section: ${u.section}`);
    });
    lines.push('');
  }

  return lines.join('\n');
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

function CustomSelect({
  options,
  value,
  onChange,
  disabled = false,
}: {
  options: { id: string; label: string; description?: string }[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
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

  const selected = options.find(option => option.id === value);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        disabled={disabled}
        className="w-full flex items-center justify-between gap-2 rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-left text-sm text-gray-700 hover:border-gray-300 disabled:cursor-default disabled:opacity-60 focus:outline-none"
      >
        <span className="min-w-0">
          <span className="block truncate font-medium">{selected?.label ?? 'Select'}</span>
          {selected?.description && (
            <span className="block truncate text-xs text-gray-400 mt-0.5">
              {selected.description}
            </span>
          )}
        </span>
        <svg className={`w-4 h-4 text-gray-400 flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute z-20 mt-2 w-full min-w-[240px] max-h-72 overflow-y-auto rounded-xl border border-gray-200 bg-white shadow-lg">
          {options.map(option => {
            const isSelected = option.id === value;
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => { onChange(option.id); setOpen(false); }}
                className={`w-full px-3 py-2.5 text-left transition-colors focus:outline-none ${
                  isSelected ? 'bg-blue-50' : 'hover:bg-gray-50'
                }`}
              >
                <span className={`block text-sm font-medium ${isSelected ? 'text-blue-700' : 'text-gray-800'}`}>
                  {option.label}
                </span>
                {option.description && (
                  <span className="block text-xs text-gray-400 mt-0.5">{option.description}</span>
                )}
              </button>
            );
          })}
        </div>
      )}
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

  const {
    grounding_results,
    faithfulness_results,
    uncited_sentences,
    citation_coverage,
    completeness,
    relevance,
  } = detail.report;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-gray-900 truncate">{detail.query}</p>
          <p className="text-xs text-gray-400 mt-0.5">
            {fmtDate(detail.generated_at)}
            {' · '}Run {detail.run_id.slice(0, 8)}
            {' · '}Eval {detail.id.slice(0, 8)}
          </p>
        </div>
        <div className="flex flex-shrink-0 items-center gap-2">
          <button
            onClick={() => downloadText(
              `${slugify(detail.query)}-eval-${new Date(detail.generated_at).toISOString().slice(0, 10)}-${detail.run_id.slice(0, 8)}-${detail.id.slice(0, 8)}.md`,
              buildEvalDetailMarkdown(detail),
            )}
            className="text-xs font-semibold text-blue-600 hover:text-blue-800 border border-blue-100 hover:border-blue-200 bg-blue-50 rounded-lg px-2.5 py-1 transition-colors"
          >
            Download .md
          </button>
        </div>
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

      {/* Source authority (absent on eval reports predating the metric) */}
      {detail.report.source_authority && (
        <div>
          <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
            Source Authority ({fmtPct(detail.report.source_authority.authority_score * 100)}
            {' · '}{detail.report.source_authority.primary_count} primary
            {' / '}{detail.report.source_authority.secondary_count} secondary
            {' / '}{detail.report.source_authority.other_count} other)
          </p>
          <div className="space-y-1.5">
            {detail.report.source_authority.verdicts.map((v, i) => (
              <div key={i} className="flex items-start gap-2 bg-white border border-gray-200 rounded-lg px-3 py-2">
                <span className={`flex-shrink-0 text-[10px] font-semibold rounded-full px-2 py-0.5 mt-0.5 ${
                  v.tier === 'primary' ? 'bg-emerald-50 text-emerald-700'
                  : v.tier === 'secondary' ? 'bg-blue-50 text-blue-700'
                  : 'bg-amber-50 text-amber-700'
                }`}>
                  {v.tier}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-gray-700 truncate">{v.domain}</p>
                  <p className="text-[10px] text-gray-400 mt-0.5">{v.reasoning}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

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

      {/* Citation coverage */}
      {citation_coverage && (
        <div>
          <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1.5">
            Citation Coverage ({fmtPct(citation_coverage.coverage_score * 100)})
          </p>
          {citation_coverage.uncited_factual_claims.length === 0 ? (
            <p className="text-xs text-gray-400">No uncited factual claims were flagged.</p>
          ) : (
            <div className="space-y-1.5">
              {citation_coverage.uncited_factual_claims.map((issue, i) => (
                <div key={i} className="bg-white border border-red-200 rounded-lg px-3 py-2">
                  <p className="text-xs text-gray-700 line-clamp-2">{issue.sentence}</p>
                  <p className="text-[10px] text-red-500 mt-0.5">{issue.reasoning}</p>
                  {issue.section && <p className="text-[10px] text-gray-400 mt-0.5">{issue.section}</p>}
                </div>
              ))}
            </div>
          )}
        </div>
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
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    { id: 'gpt-5.4', label: 'GPT-5.4', description: 'Best for complex topics' },
  ]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [runningEvalFor, setRunningEvalFor] = useState<string | null>(null);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [detail, setDetail] = useState<EvalReportDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'evaluate' | 'benchmark'>('evaluate');
  const [benchmarkQuestions, setBenchmarkQuestions] = useState('');
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState('');
  const [benchmarkStage, setBenchmarkStage] = useState('');
  const [benchmarkDetail, setBenchmarkDetail] = useState('');
  const [benchmarkPercent, setBenchmarkPercent] = useState(0);
  const detailRef = useRef<HTMLDivElement>(null);
  // Prevent an older detail request from overwriting a newer selection.
  const detailRequestRef = useRef(0);

  // Custom confirmation modal (replaces window.confirm for destructive actions)
  const [confirmDialog, setConfirmDialog] = useState<{
    title: string;
    message: string;
    confirmLabel: string;
    onConfirm: () => void;
  } | null>(null);

  // Default model used to judge faithfulness/completeness/relevance. The UI
  // displays it in benchmark mode but does not vary it per row, so the
  // comparison stays apples-to-apples.
  const [evalModel, setEvalModel] = useState('claude-haiku-4-5');
  const [researchModel, setResearchModel] = useState('gpt-5.4');
  const [advocateModel, setAdvocateModel] = useState('claude-sonnet-4-6');
  const [skepticModel, setSkepticModel] = useState('gemini-3.1-pro-preview');

  useEffect(() => {
    fetch(`${apiBase}/models`)
      .then(res => res.json())
      .then((data: ModelsResponse) => {
        if (data.options?.length) setModelOptions(data.options);
        if (data.default) setResearchModel(data.default);
        if (data.defaults?.lead) setResearchModel(data.defaults.lead);
        if (data.defaults?.eval) setEvalModel(data.defaults.eval);
        if (data.defaults?.advocate) setAdvocateModel(data.defaults.advocate);
        if (data.defaults?.skeptic) setSkepticModel(data.defaults.skeptic);
      })
      .catch(() => { /* keep the hardcoded fallback */ });
  }, [apiBase]);

  const headers = useMemo(() => ({ 'X-Client-Id': clientId }), [clientId]);

  const load = useCallback(async () => {
    setLoading(true);
    setErrorMsg('');
    try {
      const [runsRes, reportsRes] = await Promise.all([
        fetch(`${apiBase}/runs?status=done&mine=true`, { headers }),
        fetch(`${apiBase}/eval/reports`, { headers }),
      ]);
      if (!runsRes.ok || !reportsRes.ok) throw new Error('Failed to load eval data');
      setRuns(await runsRes.json());
      setReports(await reportsRes.json());
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase, headers]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (!selectedRunId && runs.length > 0) setSelectedRunId(runs[0].id);
  }, [runs, selectedRunId]);

  // Most recent eval report per run_id, for quick view/evaluate actions.
  const latestByRun = useMemo(() => {
    const map = new Map<string, EvalReportSummary>();
    for (const r of reports) {
      const existing = map.get(r.run_id);
      if (!existing || r.generated_at > existing.generated_at) map.set(r.run_id, r);
    }
    return map;
  }, [reports]);

  const benchmarkRows = useMemo(() => {
    const runsById = new Map(runs.map(run => [run.id, run]));
    const grouped = new Map<string, { reports: EvalReportSummary[]; runs: RunSummary[] }>();
    for (const report of reports) {
      const run = runsById.get(report.run_id);
      const mode = run?.stats?.mode ?? 'unknown';
      if (!['multi_agent_verified', 'debate_gap'].includes(mode)) continue;
      const group = grouped.get(mode) ?? { reports: [], runs: [] };
      group.reports.push(report);
      if (run) group.runs.push(run);
      grouped.set(mode, group);
    }

    return [...grouped.entries()]
      .map(([mode, group]) => {
        const reportCount = group.reports.length;
        const totalFindings = group.reports.reduce((s, r) => s + r.total_findings, 0);
        const ungrounded = group.reports.reduce((s, r) => s + r.ungrounded_count, 0);
        const totalCitations = group.reports.reduce((s, r) => s + r.total_citations, 0);
        const unfaithful = group.reports.reduce((s, r) => s + r.unfaithful_count, 0);
        const totalRecall = group.reports.reduce((s, r) => s + r.recall_score, 0);
        const totalRelevance = group.reports.reduce((s, r) => s + r.relevance_score, 0);
        // Only average over reports that actually carry the metric (0 = pre-metric eval)
        const authorityReports = group.reports.filter(r => r.authority_score > 0);
        const totalAuthority = authorityReports.reduce((s, r) => s + r.authority_score, 0);
        const totalCost = group.runs.reduce((s, r) => s + (r.stats?.cost_usd ?? 0), 0);
        const totalLatency = group.runs.reduce((s, r) => s + (r.stats?.elapsed_seconds ?? 0), 0);
        return {
          mode,
          reports: reportCount,
          grounding: totalFindings === 0 ? null : ((totalFindings - ungrounded) / totalFindings) * 100,
          faithfulness: totalCitations === 0 ? null : ((totalCitations - unfaithful) / totalCitations) * 100,
          completeness: reportCount === 0 ? null : (totalRecall / reportCount) * 100,
          relevance: reportCount === 0 ? null : totalRelevance / reportCount,
          authority: authorityReports.length === 0 ? null : (totalAuthority / authorityReports.length) * 100,
          avgCost: group.runs.length === 0 ? undefined : totalCost / group.runs.length,
          avgLatency: group.runs.length === 0 ? undefined : totalLatency / group.runs.length,
        };
      })
      .sort((a, b) => modeRank(a.mode) - modeRank(b.mode));
  }, [reports, runs]);

  const selectedRun = useMemo(
    () => runs.find(run => run.id === selectedRunId) ?? runs[0],
    [runs, selectedRunId],
  );
  const selectedRunEval = selectedRun ? latestByRun.get(selectedRun.id) : undefined;
  const runOptions = useMemo(
    () => runs.map(run => ({
      id: run.id,
      label: run.title || run.query || 'Untitled research',
      description: `${modeLabel(run.stats?.mode)} · ${fmtDate(run.started_at)} · ${fmtCost(run.stats?.cost_usd)}`,
    })),
    [runs],
  );

  function downloadBenchmarkMarkdown() {
    const lines = [
      '| Mode | Runs | Grounding | Faithfulness | Completeness | Relevance | Authority | Avg Cost | Avg Latency |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
      ...benchmarkRows.map(row => (
        `| ${modeLabel(row.mode)} | ${row.reports} | ${fmtPct(row.grounding)} | `
        + `${fmtPct(row.faithfulness)} | `
        + `${fmtPct(row.completeness)} | ${fmtScore(row.relevance)} | `
        + `${fmtPct(row.authority)} | `
        + `${fmtCost(row.avgCost)} | `
        + `${row.avgLatency === undefined ? '—' : `${row.avgLatency.toFixed(1)}s`} |`
      )),
    ];
    downloadText('debate-comparison.md', lines.join('\n') + '\n');
  }

  const loadDetail = useCallback(async (reportId: string, runId?: string) => {
    const requestId = ++detailRequestRef.current;
    setActiveTab('evaluate');
    if (runId) setSelectedRunId(runId);
    setSelectedReportId(reportId);
    setDetail(null);
    setDetailLoading(true);
    try {
      const res = await fetch(`${apiBase}/eval/reports/${reportId}`, { headers });
      if (!res.ok) throw new Error('Failed to load report detail');
      const loaded: EvalReportDetail = await res.json();
      if (requestId !== detailRequestRef.current) return;
      if (loaded.id !== reportId) throw new Error('Loaded eval report does not match the selection');
      setDetail(loaded);
      setTimeout(() => detailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    } catch (e) {
      if (requestId === detailRequestRef.current) setErrorMsg(String(e));
    } finally {
      if (requestId === detailRequestRef.current) setDetailLoading(false);
    }
  }, [apiBase, headers]);

  function selectRunForEval(runId: string) {
    detailRequestRef.current += 1;
    setSelectedRunId(runId);
    setSelectedReportId(null);
    setDetail(null);
    setDetailLoading(false);
  }

  async function runEval(runId: string) {
    const requestId = ++detailRequestRef.current;
    setRunningEvalFor(runId);
    setErrorMsg('');
    setSelectedReportId(null);
    setDetail(null);
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
      if (requestId !== detailRequestRef.current) return;
      setDetail(record);
      setSelectedReportId(record.id);
      await load();
    } catch (e) {
      if (requestId === detailRequestRef.current) setErrorMsg(String(e));
    } finally {
      setRunningEvalFor(null);
    }
  }

  async function startResearchRun(
    query: string,
    debate: boolean,
    onProgress: (progress: BenchmarkRunProgress) => void,
  ): Promise<string> {
    const res = await fetch(`${apiBase}/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify({
        query,
        mode: 'multi_agent_verified',
        debate,
        model: researchModel || undefined,
        advocate_model: debate ? advocateModel || undefined : undefined,
        skeptic_model: debate ? skepticModel || undefined : undefined,
      }),
    });
    if (!res.ok) throw new Error(`Research failed: HTTP ${res.status}`);

    let runId = '';
    let completed = false;
    let streamError = '';
    let initialTotal = 0;
    let initialDone = 0;
    let gapTotal = 0;
    let gapDone = 0;
    let lastPercent = 0;

    const update = (stage: string, detail: string, percent: number) => {
      lastPercent = Math.max(lastPercent, percent);
      onProgress({ stage, detail, percent: lastPercent });
    };

    await readStream(res, data => {
      const type = data.type as string;
      if (type === 'started') {
        runId = data.run_id as string;
        update('Starting research', 'Creating the research run and checking the query', 4);
      } else if (type === 'plan_thinking') {
        update('Planning research', 'The lead model is defining the research strategy', 8);
      } else if (type === 'plan') {
        initialTotal = ((data.subtasks as string[]) ?? []).length;
        update(
          'Research plan ready',
          `${initialTotal} research question${initialTotal !== 1 ? 's' : ''} will run in parallel`,
          12,
        );
      } else if (type === 'subtask_done') {
        const isGap = data.stage === 'gap';
        const question = String(data.question ?? '');
        const count = Number(data.findings_count ?? 0);
        if (isGap) {
          gapDone += 1;
          const fraction = gapTotal > 0 ? gapDone / gapTotal : 1;
          update(
            'Follow-up research',
            `${gapDone}/${gapTotal || gapDone} gap questions complete · ${count} findings · ${question}`,
            78 + fraction * 10,
          );
        } else {
          initialDone += 1;
          const fraction = initialTotal > 0 ? initialDone / initialTotal : 1;
          update(
            'Parallel research',
            `${initialDone}/${initialTotal || initialDone} questions complete · ${count} findings · ${question}`,
            14 + fraction * 38,
          );
        }
      } else if (type === 'debating') {
        update('Adversarial debate', 'Proposition and opposition are reviewing the evidence', 55);
      } else if (type === 'debate_turn') {
        const round = Number(data.round ?? 1);
        const side = data.agent === 'advocate' ? 'Proposition' : 'Opposition';
        const turnNumber = (round - 1) * 2 + (data.agent === 'advocate' ? 1 : 2);
        update(
          'Adversarial debate',
          `Round ${round} · ${side} completed`,
          55 + Math.min(turnNumber, 4) / 4 * 15,
        );
      } else if (type === 'judging') {
        update('Judging debate', 'The lead model is weighing both sides', 72);
      } else if (type === 'debate_verdict') {
        update('Debate complete', 'The verdict is ready; evidence audit is next', 75);
      } else if (type === 'evidence_auditing') {
        update(
          'Auditing evidence',
          'Checking coverage, source quality, contradictions, and missing evidence',
          debate ? 77 : 58,
        );
      } else if (type === 'evidence_audit') {
        const gaps = (data.subtasks as string[]) ?? [];
        gapTotal = gaps.length;
        update(
          gaps.length > 0 ? 'Evidence gaps found' : 'Evidence audit passed',
          gaps.length > 0
            ? `${gaps.length} targeted follow-up question${gaps.length !== 1 ? 's' : ''} planned`
            : String(data.assessment ?? 'Evidence is sufficient for synthesis'),
          gaps.length > 0 ? 78 : 82,
        );
      } else if (type === 'synthesizing') {
        update('Writing report', 'Synthesizing findings and applying citation discipline', 90);
      } else if (type === 'report') {
        update('Report ready', 'Final citation-checked report received', 96);
      } else if (type === 'clarification_needed') {
        streamError = 'Benchmark question needs clarification; use a more specific question.';
        update('Clarification required', streamError, lastPercent);
      } else if (type === 'error') {
        streamError = String(data.message ?? 'Research failed');
        update('Research failed', streamError, lastPercent);
      } else if (type === 'done') {
        runId = data.run_id as string;
        completed = true;
        update('Research complete', 'Starting report evaluation', 97);
      }
    });
    if (streamError) throw new Error(streamError);
    if (!runId) throw new Error('Research stream ended without a run id');
    if (!completed) throw new Error('Research stream ended before the run completed');
    return runId;
  }

  async function runBenchmark() {
    const questions = benchmarkQuestions
      .split('\n')
      .map(q => q.trim())
      .filter(Boolean);
    if (questions.length === 0 || benchmarkRunning) return;

    const modes = [
      { label: 'Standard Research', debate: false },
      { label: 'Debate Research', debate: true },
    ];
    const total = questions.length * modes.length;
    let done = 0;

    setBenchmarkRunning(true);
    setErrorMsg('');
    setBenchmarkStage('Preparing comparison');
    setBenchmarkDetail(`${questions.length} question${questions.length !== 1 ? 's' : ''} · ${total} total runs`);
    setBenchmarkPercent(0);
    try {
      for (const question of questions) {
        for (const mode of modes) {
          setBenchmarkProgress(`${done + 1}/${total} · ${mode.label}`);
          const runId = await startResearchRun(question, mode.debate, progress => {
            setBenchmarkStage(progress.stage);
            setBenchmarkDetail(progress.detail ? `${progress.detail} · ${question}` : question);
            setBenchmarkPercent(Math.round(((done + progress.percent / 100) / total) * 100));
          });
          setBenchmarkStage('Evaluating report');
          setBenchmarkDetail(`${mode.label} complete · scoring grounding, faithfulness, completeness, and relevance`);
          setBenchmarkPercent(Math.round(((done + 0.98) / total) * 100));
          const evalRes = await fetch(
            `${apiBase}/runs/${runId}/eval?model=${encodeURIComponent(evalModel)}`,
            { method: 'POST', headers },
          );
          if (!evalRes.ok) throw new Error(`Eval failed: HTTP ${evalRes.status}`);
          done += 1;
          setBenchmarkPercent(Math.round((done / total) * 100));
        }
      }
      setBenchmarkProgress(`Complete · ${done}/${total} runs evaluated`);
      setBenchmarkStage('Comparison complete');
      setBenchmarkDetail('Standard and debate results are ready below');
      setBenchmarkPercent(100);
      await load();
      setActiveTab('benchmark');
    } catch (e) {
      setErrorMsg(String(e));
      setBenchmarkStage('Comparison stopped');
      setBenchmarkDetail(String(e));
    } finally {
      setBenchmarkRunning(false);
    }
  }

  async function deleteRun(runId: string) {
    setDeletingRunId(runId);
    setErrorMsg('');
    try {
      const res = await fetch(`${apiBase}/runs/${runId}`, { method: 'DELETE', headers });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (selectedReportId && latestByRun.get(runId)?.id === selectedReportId) {
        detailRequestRef.current += 1;
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

      <div className="inline-flex rounded-xl border border-gray-200 bg-white p-1 shadow-sm">
        {[
          ['evaluate', 'Evaluate Reports'],
          ['benchmark', 'Compare Debate'],
        ].map(([id, label]) => (
          <button
            key={id}
            onClick={() => setActiveTab(id as 'evaluate' | 'benchmark')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === id ? 'bg-gray-900 text-white' : 'text-gray-500 hover:text-gray-800'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32 text-sm text-gray-400 gap-2">
          <Spinner /> Loading…
        </div>
      ) : (
        <>
          {activeTab === 'benchmark' && (
            <>
            <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
              <div className="flex items-start justify-between gap-3 flex-wrap">
                <div>
                  <p className="text-sm font-semibold text-gray-900">Standard vs Debate</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Runs each question once normally and once with experimental debate, then compares the results.
                  </p>
                </div>
                <button
                  onClick={runBenchmark}
                  disabled={benchmarkRunning || !benchmarkQuestions.trim()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-gray-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-gray-800 disabled:opacity-50 disabled:cursor-default transition-colors"
                >
                  {benchmarkRunning && <Spinner />}
                  {benchmarkRunning ? 'Running…' : 'Compare Debate'}
                </button>
              </div>

              <div className="grid gap-2 sm:grid-cols-2 mt-4">
                {[
                  ['Standard Research', 'Parallel research, evidence audit, and citation checks'],
                  ['Debate Research', 'The same pipeline with adversarial review before the audit'],
                ].map(([label, text]) => (
                  <div key={label} className="rounded-xl border border-gray-200 bg-gray-50 px-3 py-2">
                    <p className="text-xs font-bold text-gray-900">{label}</p>
                    <p className="text-[11px] text-gray-500 mt-0.5 leading-snug">{text}</p>
                  </div>
                ))}
              </div>

              <div className="grid gap-4 lg:grid-cols-[1fr_260px] mt-4">
                <div>
                  <textarea
                    value={benchmarkQuestions}
                    onChange={e => setBenchmarkQuestions(e.target.value)}
                    disabled={benchmarkRunning}
                    rows={5}
                    placeholder="Add one benchmark question per line..."
                    className="w-full rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-700 focus:outline-none disabled:opacity-60"
                  />
                  <div className="flex flex-wrap gap-2 mt-2">
                    {[
                      'How is Singapore regulating AI governance and model risk in 2026?',
                      'What obligations does the EU AI Act create for high-risk AI systems?',
                      'How are the US, EU, and UK regulating frontier AI models?',
                    ].map(question => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => setBenchmarkQuestions(prev =>
                          prev.includes(question)
                            ? prev
                            : `${prev}${prev.trim() ? '\n' : ''}${question}`,
                        )}
                        disabled={benchmarkRunning}
                        className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-600 hover:border-blue-300 hover:text-blue-600 disabled:opacity-50 transition-colors"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1">
                      Research Model
                    </p>
                    <CustomSelect
                      options={modelOptions}
                      value={researchModel}
                      onChange={setResearchModel}
                      disabled={benchmarkRunning}
                    />
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1">
                      Debate Advocate
                    </p>
                    <CustomSelect
                      options={modelOptions}
                      value={advocateModel}
                      onChange={setAdvocateModel}
                      disabled={benchmarkRunning}
                    />
                  </div>
                  <div>
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold mb-1">
                      Debate Skeptic
                    </p>
                    <CustomSelect
                      options={modelOptions}
                      value={skepticModel}
                      onChange={setSkepticModel}
                      disabled={benchmarkRunning}
                    />
                  </div>
                  <div className="rounded-xl border border-gray-200 bg-gray-50 px-3 py-2">
                    <p className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold">
                      Eval Judge
                    </p>
                    <p className="text-sm font-medium text-gray-700 mt-0.5">
                      {modelOptions.find(model => model.id === evalModel)?.label ?? evalModel}
                    </p>
                  </div>
                </div>
              </div>
              {benchmarkProgress && (
                <div className="mt-4 rounded-xl border border-blue-100 bg-blue-50/60 px-3.5 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-xs font-semibold text-gray-800">
                        {benchmarkRunning && <span className="inline-block mr-1.5"><Spinner /></span>}
                        {benchmarkStage || 'Preparing comparison'}
                      </p>
                      <p className="text-[11px] text-gray-500 mt-0.5 truncate">
                        {benchmarkProgress}{benchmarkDetail ? ` · ${benchmarkDetail}` : ''}
                      </p>
                    </div>
                    <span className="text-xs font-bold tabular-nums text-blue-700">
                      {benchmarkPercent}%
                    </span>
                  </div>
                  <div className="mt-2.5 h-1.5 rounded-full bg-blue-100 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-blue-600 transition-[width] duration-500 ease-out"
                      style={{ width: `${benchmarkPercent}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
              <div className="px-4 pt-4 pb-3 border-b border-gray-100 flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-gray-900">Debate Comparison Results</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Averages across evaluated standard and debate runs.
                  </p>
                </div>
                {benchmarkRows.length > 0 && (
                  <button
                    onClick={downloadBenchmarkMarkdown}
                    className="text-xs font-semibold text-blue-600 hover:text-blue-800 border border-blue-100 hover:border-blue-200 bg-blue-50 rounded-lg px-2.5 py-1 transition-colors whitespace-nowrap"
                  >
                    Download .md
                  </button>
                )}
              </div>
              {benchmarkRows.length === 0 ? (
                <p className="text-sm text-gray-400 px-4 py-4">
                  Run and evaluate at least one research task to populate the benchmark table.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-[11px] uppercase tracking-wide text-gray-400 font-semibold border-b border-gray-200">
                        <th className="text-left px-4 py-2 font-semibold">Mode</th>
                        <th className="text-right px-4 py-2 font-semibold">Runs</th>
                        <th className="text-right px-4 py-2 font-semibold">Grounding</th>
                        <th className="text-right px-4 py-2 font-semibold">Faithfulness</th>
                        <th className="text-right px-4 py-2 font-semibold">Completeness</th>
                        <th className="text-right px-4 py-2 font-semibold">Relevance</th>
                        <th className="text-right px-4 py-2 font-semibold">Authority</th>
                        <th className="text-right px-4 py-2 font-semibold">Avg Cost</th>
                        <th className="text-right px-4 py-2 font-semibold">Avg Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarkRows.map(row => (
                        <tr key={row.mode} className="border-b border-gray-100 last:border-0">
                          <td className="px-4 py-2.5 font-semibold text-gray-800 whitespace-nowrap">
                            {modeLabel(row.mode)}
                          </td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{row.reports}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{fmtPct(row.grounding)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{fmtPct(row.faithfulness)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{fmtPct(row.completeness)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{fmtScore(row.relevance)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{fmtPct(row.authority)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">{fmtCost(row.avgCost)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500">
                            {row.avgLatency === undefined ? '—' : `${row.avgLatency.toFixed(1)}s`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            </>
          )}

          {/* Evaluate reports */}
          {activeTab === 'evaluate' && (
          <>
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900">Evaluate a Report</p>
            <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_auto]">
              <CustomSelect
                options={runOptions.length ? runOptions : [{
                  id: '',
                  label: 'No completed runs yet',
                  description: 'Start a research run first',
                }]}
                value={selectedRun?.id ?? ''}
                onChange={selectRunForEval}
                disabled={runs.length === 0}
              />
              <div className="flex items-center gap-2 justify-end">
                {selectedRunEval ? (
                  <button
                    onClick={() => loadDetail(selectedRunEval.id)}
                    className="rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700 transition-colors"
                  >
                    View Eval
                  </button>
                ) : (
                  <button
                    onClick={() => selectedRun && runEval(selectedRun.id)}
                    disabled={!selectedRun || runningEvalFor === selectedRun?.id}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-default transition-colors"
                  >
                    {runningEvalFor === selectedRun?.id && <Spinner />}
                    {runningEvalFor === selectedRun?.id ? 'Running…' : 'Run Eval'}
                  </button>
                )}
                {selectedRun && (
                  <button
                    onClick={() => confirmDeleteRun(selectedRun.id)}
                    disabled={deletingRunId === selectedRun.id}
                    className="rounded-lg border border-gray-200 px-3 py-2 text-xs font-semibold text-gray-500 hover:text-red-600 disabled:opacity-50 transition-colors"
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
            {selectedRunEval && (
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mt-4">
                {[
                  ['Grounding', fmtPct(groundingRate(selectedRunEval))],
                  ['Faithfulness', fmtPct(faithfulnessRate(selectedRunEval))],
                  ['Completeness', fmtPct(selectedRunEval.recall_score * 100)],
                  ['Relevance', fmtScore(selectedRunEval.relevance_score)],
                  // 0 means "evaluated before the authority metric existed", not a real score
                  ['Authority', selectedRunEval.authority_score > 0 ? fmtPct(selectedRunEval.authority_score * 100) : '—'],
                ].map(([label, value]) => (
                  <div key={label} className="bg-gray-50 border border-gray-200 rounded-xl px-3 py-2">
                    <p className="text-[10px] uppercase tracking-wide text-gray-400 font-semibold">
                      {label}
                    </p>
                    <p className="text-base font-bold text-gray-900 mt-0.5">{value}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

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
                      <th className="text-left px-4 py-2 font-semibold">Mode</th>
                      <th className="text-left px-4 py-2 font-semibold">Started</th>
                      <th className="text-right px-4 py-2 font-semibold">Cost</th>
                      <th className="text-right px-4 py-2 font-semibold">Tokens</th>
                      <th className="text-right px-4 py-2 font-semibold">Elapsed</th>
                      <th className="text-right px-4 py-2 font-semibold">Authority</th>
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
                          <td className="px-4 py-2.5 text-gray-500 whitespace-nowrap">
                            {modeLabel(run.stats?.mode)}
                          </td>
                          <td className="px-4 py-2.5 text-gray-500 whitespace-nowrap">{fmtDate(run.started_at)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{fmtCost(run.stats?.cost_usd)}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{run.stats?.total_tokens?.toLocaleString() ?? '—'}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">{run.stats?.elapsed_seconds !== undefined ? `${run.stats.elapsed_seconds}s` : '—'}</td>
                          <td className="px-4 py-2.5 text-right text-gray-500 whitespace-nowrap">
                            {latest && latest.authority_score > 0 ? fmtPct(latest.authority_score * 100) : '—'}
                          </td>
                          <td className="px-4 py-2.5 text-right whitespace-nowrap">
                            <div className="flex items-center justify-end gap-3">
                              {latest ? (
                                <button
                                  onClick={() => loadDetail(latest.id, run.id)}
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

          {/* Metric definitions — what each eval score actually measures */}
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900 mb-2">What the metrics measure</p>
            <dl className="grid gap-x-6 gap-y-1.5 sm:grid-cols-2 text-xs">
              {[
                ['Grounding', 'Every finding’s quoted evidence is re-fetched from its source URL and matched against the live page — citations point at real text, checked without an LLM.'],
                ['Faithfulness', 'An independent judge model verifies each cited report sentence against the findings behind its citation — the report doesn’t overstate its sources.'],
                ['Completeness', 'The judge generates the subtopics a good answer should cover, then checks the report against them. The rubric is regenerated per eval, so scores vary a few points between evals of the same report.'],
                ['Relevance', '1–5: does the report answer the question that was actually asked, rather than a nearby easier one.'],
                ['Authority', 'Unique cited domains tiered primary (regulators, courts, standards bodies) / secondary (major law firms, news, academia) / other (vendor blogs, content sites), weighted 1.0/0.6/0.2. Complements grounding: a quote can be accurate but from a weak source.'],
              ].map(([term, def]) => (
                <div key={term} className="flex gap-2">
                  <dt className="flex-shrink-0 w-24 font-semibold text-gray-600">{term}</dt>
                  <dd className="text-gray-400">{def}</dd>
                </div>
              ))}
            </dl>
          </div>
          </>
          )}

          {/* Detail panel */}
          {activeTab === 'evaluate' && (
          <div ref={detailRef} className="bg-white border border-gray-200 rounded-2xl shadow-sm p-4">
            <p className="text-sm font-semibold text-gray-900 mb-3">Report Detail</p>
            <DetailPanel
              detail={detail?.id === selectedReportId ? detail : null}
              loading={detailLoading}
            />
          </div>
          )}

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
