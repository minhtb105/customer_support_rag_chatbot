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

export async function queryRag(query: string, top_k = 5, user_id = "default_user") {
  const res = await authFetch(`${API}/v1/query`, {
    method: "POST",
    body: JSON.stringify({ query, top_k, user_id, include_audit: true }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`query failed ${res.status}: ${txt.slice(0,300)}`);
  }
  return res.json();
}

// SSE streaming for POST /v1/query/stream — yields {event, data} via callback
export async function queryRagStream(
  query: string,
  opts: { top_k?: number; user_id?: string; onEvent?: (ev: string, data: any) => void; onToken?: (delta: string) => void; signal?: AbortSignal } = {}
) {
  const { top_k = 5, user_id = "default_user", onEvent, onToken, signal } = opts;
  const res = await authFetch(`${API}/v1/query/stream`, {
    method: "POST",
    body: JSON.stringify({ query, top_k, user_id, include_audit: true }),
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
  const res = await authFetch(`${API}/v1/glucose`, { method: "POST", body: JSON.stringify(payload) });
  if (!res.ok) throw new Error(`glucose log failed ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function getGlucose(user_id: string, limit = 50, days?: number) {
  const url = new URL(`${API}/v1/glucose/${encodeURIComponent(user_id)}`);
  url.searchParams.set("limit", String(limit));
  if (days) url.searchParams.set("days", String(days));
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`get glucose failed ${res.status}`);
  return res.json();
}
export async function saveGlucoseFollowup(payload: { user_id: string; text: string; related_log_id?: number }) {
  const res = await authFetch(`${API}/v1/glucose/followup`, { method: "POST", body: JSON.stringify(payload) });
  if (!res.ok) throw new Error(`followup failed ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function generateSoap(user_id: string, days = 14, disease: string = "diabetes") {
  const res = await authFetch(`${API}/v1/soap/generate`, { method: "POST", body: JSON.stringify({ user_id, days, language: "vi", disease }) });
  if (!res.ok) throw new Error(`soap failed ${res.status}`);
  return res.json();
}
export async function generateSoapMarkdown(user_id: string, days = 14, disease: string = "diabetes") {
  const res = await authFetch(`${API}/v1/soap/generate/markdown`, { method: "POST", body: JSON.stringify({ user_id, days, language: "vi", disease }) });
  if (!res.ok) throw new Error(`soap md failed ${res.status}`);
  return res.text();
}
export async function getHealth() {
  const res = await fetch(`${API}/v1/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("health failed");
  return res.json();
}
export async function getGuidelinesStatus() {
  const res = await fetch(`${API}/v1/guidelines/status`, { cache: "no-store" });
  if (!res.ok) throw new Error("guidelines failed");
  return res.json();
}
// Reviews & notifications
export async function getReviews(status?: string) {
  const url = new URL(`${API}/v1/reviews`);
  if (status) url.searchParams.set("status", status);
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`get reviews failed ${res.status}`);
  return res.json();
}
export async function getMyReviews() {
  const res = await authFetch(`${API}/v1/reviews/me`, { method: "GET" });
  if (!res.ok) throw new Error(`my reviews failed`);
  return res.json();
}
export async function getReviewDetail(id: string) {
  const res = await authFetch(`${API}/v1/reviews/${id}`, { method: "GET" });
  if (!res.ok) throw new Error(`review ${res.status}`);
  return res.json();
}
export async function decideReview(id: string, body: { decision: string; final_answer?: string; expert_notes?: string }) {
  const res = await authFetch(`${API}/v1/reviews/${id}/decide`, { method: "POST", body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`decide failed ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function getNotifications() {
  const res = await authFetch(`${API}/v1/auth/notifications`, { method: "GET" });
  if (!res.ok) throw new Error(`notif failed`);
  return res.json();
}
export async function getQueryHistory() {
  const res = await authFetch(`${API}/v1/query/history`, { method: "GET" });
  if (!res.ok) throw new Error(`history failed`);
  return res.json();
}
export async function adminListUsers(role?: string) {
  const url = new URL(`${API}/v1/auth/users`);
  if (role) url.searchParams.set("role", role);
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`list users ${res.status}`);
  return res.json();
}
export async function adminUpdateUser(id: string, data: any) {
  const res = await authFetch(`${API}/v1/auth/users/${id}`, { method: "PATCH", body: JSON.stringify(data) });
  if (!res.ok) throw new Error(`update user ${res.status}`);
  return res.json();
}
export async function adminVerifyExpert(id: string) {
  const res = await authFetch(`${API}/v1/auth/users/${id}/verify`, { method: "POST" });
  if (!res.ok) throw new Error(`verify ${res.status}`);
  return res.json();
}
// Tracing (admin)
export async function adminListTraces(params: any = {}) {
  const url = new URL(`${API}/v1/admin/traces`);
  Object.entries(params).forEach(([k,v])=>{ if(v!==undefined && v!==null && v!=="") url.searchParams.set(k, String(v)); });
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`traces ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminGetTrace(id: string) {
  const res = await authFetch(`${API}/v1/admin/traces/${id}`, { method: "GET" });
  if (!res.ok) throw new Error(`trace ${res.status}`);
  return res.json();
}
export async function adminTriggerRagas(id: string) {
  const res = await authFetch(`${API}/v1/admin/traces/${id}/ragas`, { method: "POST" });
  if (!res.ok) throw new Error(`ragas ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminListCollections() {
  const res = await authFetch(`${API}/v1/admin/collections`, { method: "GET" });
  if (!res.ok) throw new Error(`collections ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminListCollectionChunks(strategy: string, params: { page?: number; limit?: number; dataset?: string; q?: string } = {}) {
  const url = new URL(`${API}/v1/admin/collections/${encodeURIComponent(strategy)}/chunks`);
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v)); });
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`chunks ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminListUserFacts(user_id: string, params: { page?: number; limit?: number; fact_type?: string; q?: string } = {}) {
  const url = new URL(`${API}/v1/admin/memory/facts`);
  url.searchParams.set("user_id", user_id);
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v)); });
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`memory facts ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminListPrompts(tone?: string) {
  const url = new URL(`${API}/v1/admin/prompts`);
  if (tone) url.searchParams.set("tone", tone);
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`prompts ${res.status}`);
  return res.json();
}
export async function adminGetPromptTone(tone: string) {
  const res = await authFetch(`${API}/v1/admin/prompts/${tone}`, { method: "GET" });
  if (!res.ok) throw new Error(`prompt ${res.status}`);
  return res.json();
}
export async function adminCreatePromptDraft(tone: string, text: string, description?: string) {
  const res = await authFetch(`${API}/v1/admin/prompts/draft`, { method: "POST", body: JSON.stringify({ tone, text, description }) });
  if (!res.ok) throw new Error(`draft ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminApprovePrompt(tone: string, version: string) {
  const res = await authFetch(`${API}/v1/admin/prompts/${tone}/approve/${version}`, { method: "POST" });
  if (!res.ok) throw new Error(`approve ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function adminRejectPrompt(tone: string, version: string) {
  const res = await authFetch(`${API}/v1/admin/prompts/${tone}/reject/${version}`, { method: "POST" });
  if (!res.ok) throw new Error(`reject ${res.status}`);
  return res.json();
}
export async function adminDryRun(tone: string, text: string, queries: string[]) {
  const res = await authFetch(`${API}/v1/admin/prompts/dry-run`, { method: "POST", body: JSON.stringify({ tone, text, queries }) });
  if (!res.ok) throw new Error(`dry-run ${res.status}: ${await res.text()}`);
  return res.json();
}

// Monitors
export async function getMonitorsStatus() {
  const res = await authFetch(`${API}/v1/monitors/status`, { method: "GET" });
  if (!res.ok) throw new Error(`monitors status ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function getMonitorSources() {
  const res = await authFetch(`${API}/v1/monitors/sources`, { method: "GET" });
  if (!res.ok) throw new Error(`monitor sources ${res.status}`);
  return res.json();
}
export async function getMonitorRuns(source_key?: string, limit = 20) {
  const url = new URL(`${API}/v1/monitors/runs`);
  if (source_key) url.searchParams.set("source_key", source_key);
  url.searchParams.set("limit", String(limit));
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`monitor runs ${res.status}`);
  return res.json();
}
export async function triggerGuidelineCheck(source_key?: string, force = true) {
  const url = new URL(`${API}/v1/monitors/check/guidelines`);
  if (source_key) url.searchParams.set("source_key", source_key);
  url.searchParams.set("force", String(force));
  const res = await authFetch(url.toString(), { method: "POST" });
  if (!res.ok) throw new Error(`trigger guideline ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function listGuidelines(params: { status?: string; source?: string; limit?: number; offset?: number } = {}) {
  const url = new URL(`${API}/v1/monitors/guidelines`);
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v)); });
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`list guidelines ${res.status}`);
  return res.json();
}
export async function getGuideline(gid: string) {
  const res = await authFetch(`${API}/v1/monitors/guidelines/${encodeURIComponent(gid)}`, { method: "GET" });
  if (!res.ok) throw new Error(`get guideline ${res.status}`);
  return res.json();
}
export async function decideGuideline(gid: string, decision: "approved" | "rejected", notes?: string) {
  const res = await authFetch(`${API}/v1/monitors/guidelines/${encodeURIComponent(gid)}/decision`, { method: "POST", body: JSON.stringify({ decision, notes }) });
  if (!res.ok) throw new Error(`decide guideline ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function triggerSafetyCheck(source: "fda" | "byt" | "all" = "all") {
  const res = await authFetch(`${API}/v1/monitors/check/safety?source=${source}`, { method: "POST" });
  if (!res.ok) throw new Error(`trigger safety ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function listAlerts(params: { status?: string; severity?: string; source?: string; limit?: number; offset?: number } = {}) {
  const url = new URL(`${API}/v1/monitors/alerts`);
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v)); });
  const res = await authFetch(url.toString(), { method: "GET" });
  if (!res.ok) throw new Error(`list alerts ${res.status}`);
  return res.json();
}
export async function getAlert(aid: string) {
  const res = await authFetch(`${API}/v1/monitors/alerts/${encodeURIComponent(aid)}`, { method: "GET" });
  if (!res.ok) throw new Error(`get alert ${res.status}`);
  return res.json();
}
export async function decideAlert(aid: string, decision: "approved" | "dismissed", notes?: string) {
  const res = await authFetch(`${API}/v1/monitors/alerts/${encodeURIComponent(aid)}/decision`, { method: "POST", body: JSON.stringify({ decision, notes }) });
  if (!res.ok) throw new Error(`decide alert ${res.status}: ${await res.text()}`);
  return res.json();
}
export async function getMonitorsStats() {
  const res = await authFetch(`${API}/v1/monitors/stats`, { method: "GET" });
  if (!res.ok) throw new Error(`monitors stats ${res.status}`);
  return res.json();
}

export async function getDoctorPatients() {
  const res = await authFetch(`${API}/v1/doctor/patients`, { method: "GET" });
  if (!res.ok) throw new Error(`doctor patients ${res.status}`);
  return res.json();
}
export const API_BASE = API;
