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

export async function logGlucose(payload: { user_id: string; value_mgdl: number; context: string; notes?: string; measured_at?: string }) {
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

export const API_BASE = API;
