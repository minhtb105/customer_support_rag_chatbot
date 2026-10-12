// Central API helpers for FastAPI WHO-RAG backend (auth via httpOnly cookie)
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function authFetch(url: string, opts: RequestInit = {}) {
  const res = await fetch(url, { ...opts, credentials: "include" as const, headers: { "Content-Type": "application/json", ...(opts.headers as any) } });
  if (res.status === 401) {
    // try refresh once
    const r = await fetch(`${API}/v1/auth/refresh`, { method: "POST", credentials: "include" as const });
    if (r.ok) {
      return fetch(url, { ...opts, credentials: "include" as const, headers: { "Content-Type": "application/json", ...(opts.headers as any) } });
    }
  }
  return res;
}

// Shared GET/POST over authFetch: builds query params, throws with slice(0,300).
async function apiGet<T = any>(path: string, params?: Record<string, any>, label?: string): Promise<T> {
  const url = new URL(`${API}${path}`);
  Object.entries(params || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v)); });
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`${label || `GET ${path}`} failed ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

async function apiPost<T = any>(path: string, body?: any, label?: string, method = "POST"): Promise<T> {
  const res = await authFetch(`${API}${path}`, { method, body: body !== undefined ? JSON.stringify(body) : undefined });
  if (!res.ok) throw new Error(`${label || `POST ${path}`} failed ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

async function apiPostText(path: string, body?: any, label?: string): Promise<string> {
  const res = await authFetch(`${API}${path}`, { method: "POST", body: body !== undefined ? JSON.stringify(body) : undefined });
  if (!res.ok) throw new Error(`${label || `POST ${path}`} failed ${res.status}: ${(await res.text()).slice(0, 300)}`);
  return res.text();
}

// DB stores failed_metrics as JSON in most rows, but some legacy rows hold
// a Python-repr string — parse defensively so one bad row never crashes a list.
export function parseFailedMetrics(v: any): string[] {
  if (!v) return [];
  if (Array.isArray(v)) return v.map(String);
  if (typeof v === "string") {
    try {
      const p = JSON.parse(v);
      return Array.isArray(p) ? p.map(String) : [String(p)];
    } catch {
      return v.replace(/^\[|\]$/g, "").split(",").map((s) => s.replace(/^['"\s]+|['"\s]+$/g, "")).filter(Boolean);
    }
  }
  return [String(v)];
}

// Pretty-print a JSON blob, or return the raw text when it is not valid JSON.
export function formatRawJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return String(text);
  }
}

// m5: một định dạng giờ duy nhất cho UI (vi-VN, Asia/Ho_Chi_Minh).
// Nhận ISO "2026-10-07T03:21" hoặc có giây/múi giờ; parse lỗi -> trả chuỗi gốc.
export function fmtDT(iso: any): string {
  if (!iso) return "—";
  const s = String(iso);
  const d = new Date(s.length <= 16 ? s + ":00" : s);
  if (isNaN(d.getTime())) return s;
  return d.toLocaleString("vi-VN", {
    timeZone: "Asia/Ho_Chi_Minh",
    day: "2-digit", month: "2-digit", year: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

// Shared RAG request body (queryRag + queryRagStream, same contract).
function ragBody(query: string, top_k = 5, user_id = "default_user") {
  return { query, top_k, user_id, include_audit: true };
}

export async function queryRag(query: string, top_k = 5, user_id = "default_user", language = "vi") {
  return apiPost("/v1/query", { ...ragBody(query, top_k, user_id), language }, "query");
}

// SSE streaming for POST /v1/query/stream — yields {event, data} via callback
export async function queryRagStream(
  query: string,
  opts: { top_k?: number; user_id?: string; onEvent?: (ev: string, data: any) => void; onToken?: (delta: string) => void; signal?: AbortSignal } = {}
) {
  const { top_k = 5, user_id = "default_user", onEvent, onToken, signal } = opts;
  const res = await authFetch(`${API}/v1/query/stream`, {
    method: "POST",
    body: JSON.stringify(ragBody(query, top_k, user_id)),
    headers: { Accept: "text/event-stream" },
    signal,
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`query/stream failed ${res.status}: ${txt.slice(0,300)}`);
  }
  if (!res.body) throw new Error("No response body for SSE stream");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const parseAndEmit = (chunk: string) => {
    buffer += chunk;
    // SSE frames are separated by \n\n
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (!raw.trim()) continue;
      let ev = "message";
      let dataStr = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      let data: any = dataStr;
      try {
        data = JSON.parse(dataStr);
      } catch {}
      onEvent?.(ev, data);
      if (ev === "token" && data?.delta) onToken?.(data.delta);
    }
  };
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parseAndEmit(decoder.decode(value, { stream: true }));
  }
  // flush remaining
  if (buffer.trim()) parseAndEmit("\n\n");
  return;
}

export interface GlucoseAnomaly { type: "spike" | "trend" | "none"; direction?: string; reason: string; }
export interface GlucoseLogOut { id: number; user_id: string; value_mgdl: number; measured_at: string; context: string; notes?: string; classification: string; message: string; anomaly?: GlucoseAnomaly | null; follow_up_questions?: string[]; }
export async function logGlucose(payload: { user_id: string; value_mgdl: number; context: string; notes?: string; measured_at?: string }): Promise<GlucoseLogOut> {
  return apiPost("/v1/glucose", payload, "glucose log");
}
export async function getGlucose(user_id: string, limit = 50, days?: number) {
  return apiGet(`/v1/glucose/${encodeURIComponent(user_id)}`, { limit, days }, "get glucose");
}
export async function saveGlucoseFollowup(payload: { user_id: string; text: string; related_log_id?: number }) {
  return apiPost("/v1/glucose/followup", payload, "followup");
}
// Shared SOAP body (JSON + Markdown share user_id/days/language/disease).
function soapBody(user_id: string, days = 14, disease: string = "diabetes") {
  return { user_id, days, language: "vi", disease };
}
export async function generateSoap(user_id: string, days = 14, disease: string = "diabetes") {
  return apiPost("/v1/soap/generate", soapBody(user_id, days, disease), "soap");
}
export async function generateSoapMarkdown(user_id: string, days = 14, disease: string = "diabetes") {
  return apiPostText("/v1/soap/generate/markdown", soapBody(user_id, days, disease), "soap md");
}
export async function getHealth() {
  const res = await fetch(`${API}/v1/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("health failed");
  return res.json();
}

// P3: tra cứu dinh dưỡng cấu trúc (không LLM) — món ăn VN cho tiểu đường.
export async function lookupNutrition(query: string, profile: "mild" | "severe" | "both" = "both", top_k = 3) {
  return apiPost("/v1/nutrition/lookup", { query, profile, top_k }, "lookup");
}
export async function getGuidelinesStatus() {
  const res = await fetch(`${API}/v1/guidelines/status`, { cache: "no-store" });
  if (!res.ok) throw new Error("guidelines failed");
  return res.json();
}
// Reviews & notifications (shared /v1/reviews base).
function reviewsPath(me = false) {
  return me ? "/v1/reviews/me" : "/v1/reviews";
}
export async function getReviews(status?: string) {
  return apiGet(reviewsPath(), { status }, "get reviews");
}
export async function getMyReviews() {
  return apiGet(reviewsPath(true), undefined, "my reviews");
}
export async function getReviewDetail(id: string) {
  return apiGet(`/v1/reviews/${id}`, undefined, "review");
}
export async function decideReview(id: string, body: { decision: string; final_answer?: string; expert_notes?: string }) {
  return apiPost(`/v1/reviews/${id}/decide`, body, "decide");
}
export async function getNotifications() {
  return apiGet("/v1/auth/notifications", undefined, "notif");
}
export async function getQueryHistory() {
  return apiGet("/v1/query/history", undefined, "history");
}
export async function adminListUsers(role?: string) {
  return apiGet("/v1/auth/users", { role }, "list users");
}
export async function adminUpdateUser(id: string, data: any) {
  return apiPost(`/v1/auth/users/${id}`, data, "update user", "PATCH");
}
export async function adminVerifyExpert(id: string) {
  return apiPost(`/v1/auth/users/${id}/verify`, undefined, "verify");
}
// Tracing (admin)
export async function adminListTraces(params: any = {}) {
  return apiGet("/v1/admin/traces", params, "traces");
}
export async function adminGetTrace(id: string) {
  return apiGet(`/v1/admin/traces/${id}`, undefined, "trace");
}
export async function adminTriggerRagas(id: string) {
  return apiPost(`/v1/admin/traces/${id}/ragas`, undefined, "ragas");
}
export async function adminListCollections() {
  return apiGet("/v1/admin/collections", undefined, "collections");
}
export async function adminListCollectionChunks(strategy: string, params: { page?: number; limit?: number; dataset?: string; q?: string } = {}) {
  return apiGet(`/v1/admin/collections/${encodeURIComponent(strategy)}/chunks`, params, "chunks");
}
export async function adminListUserFacts(user_id: string, params: { page?: number; limit?: number; fact_type?: string; q?: string } = {}) {
  return apiGet("/v1/admin/memory/facts", { user_id, ...params }, "memory facts");
}
export async function adminListPrompts(tone?: string) {
  return apiGet("/v1/admin/prompts", { tone }, "prompts");
}
export async function adminGetPromptTone(tone: string) {
  return apiGet(`/v1/admin/prompts/${tone}`, undefined, "prompt");
}
export async function adminCreatePromptDraft(tone: string, text: string, description?: string) {
  return apiPost("/v1/admin/prompts/draft", { tone, text, description }, "draft");
}
export async function adminApprovePrompt(tone: string, version: string) {
  return apiPost(`/v1/admin/prompts/${tone}/approve/${version}`, undefined, "approve");
}
export async function adminRejectPrompt(tone: string, version: string) {
  return apiPost(`/v1/admin/prompts/${tone}/reject/${version}`, undefined, "reject");
}
export async function adminDryRun(tone: string, text: string, queries: string[]) {
  return apiPost("/v1/admin/prompts/dry-run", { tone, text, queries }, "dry-run");
}

// Monitors
export async function getMonitorsStatus() {
  return apiGet("/v1/monitors/status", undefined, "monitors status");
}
export async function getMonitorSources() {
  return apiGet("/v1/monitors/sources", undefined, "monitor sources");
}
export async function getMonitorRuns(source_key?: string, limit = 20) {
  return apiGet("/v1/monitors/runs", { source_key, limit }, "monitor runs");
}
export async function triggerGuidelineCheck(source_key?: string, force = true) {
  return apiPost(`/v1/monitors/check/guidelines?source_key=${encodeURIComponent(source_key || "")}&force=${String(force)}`, undefined, "trigger guideline");
}
export async function listGuidelines(params: { status?: string; source?: string; limit?: number; offset?: number } = {}) {
  return apiGet("/v1/monitors/guidelines", params, "list guidelines");
}
export async function getGuideline(gid: string) {
  return apiGet(`/v1/monitors/guidelines/${encodeURIComponent(gid)}`, undefined, "get guideline");
}
export async function decideGuideline(gid: string, decision: "approved" | "rejected", notes?: string) {
  return apiPost(`/v1/monitors/guidelines/${encodeURIComponent(gid)}/decision`, { decision, notes }, "decide guideline");
}
export async function triggerSafetyCheck(source: "fda" | "byt" | "all" = "all") {
  return apiPost(`/v1/monitors/check/safety?source=${source}`, undefined, "trigger safety");
}
export async function listAlerts(params: { status?: string; severity?: string; source?: string; limit?: number; offset?: number } = {}) {
  return apiGet("/v1/monitors/alerts", params, "list alerts");
}
export async function getAlert(aid: string) {
  return apiGet(`/v1/monitors/alerts/${encodeURIComponent(aid)}`, undefined, "get alert");
}
export async function decideAlert(aid: string, decision: "approved" | "dismissed", notes?: string) {
  return apiPost(`/v1/monitors/alerts/${encodeURIComponent(aid)}/decision`, { decision, notes }, "decide alert");
}
export async function getMonitorsStats() {
  return apiGet("/v1/monitors/stats", undefined, "monitors stats");
}

export async function getDoctorPatients(limit = 100, offset = 0) {
  return apiGet("/v1/doctor/patients", { limit, offset }, "doctor patients");
}
export async function adminListTriageEvents(params: { page?: number; limit?: number; specialty?: string; emergency?: boolean; q?: string } = {}) {
  return apiGet("/v1/admin/triage/events", params, "triage events");
}
export const API_BASE = API;

// Micro-curriculum (learn): lessons + quiz + adaptive + 7-day today.
export async function getLessons(userId: string) {
  return apiGet("/v1/lessons", { user_id: userId }, "lessons");
}
export async function getLesson(lessonId: string, userId: string) {
  return apiGet(`/v1/lessons/${encodeURIComponent(lessonId)}`, { user_id: userId }, "lesson");
}
export async function submitLessonQuiz(lessonId: string, userId: string, answer: number | string) {
  const body = typeof answer === "string" ? { answer_text: answer } : { answer };
  return apiPost(`/v1/lessons/${encodeURIComponent(lessonId)}/quiz?user_id=${encodeURIComponent(userId)}`, body, "quiz");
}
export async function getNextLesson(userId: string) {
  return apiGet("/v1/lessons/next", { user_id: userId }, "next lesson");
}
export async function getTodayLesson(userId: string) {
  return apiGet("/v1/lessons/today", { user_id: userId }, "today lesson");
}
export async function saveLearnOnboarding(userId: string, data: { age?: number; meds?: string; has_meter?: number }) {
  return apiPost(`/v1/lessons/onboarding?user_id=${encodeURIComponent(userId)}`, data, "learn onboarding");
}
