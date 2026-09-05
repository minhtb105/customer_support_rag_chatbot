// Central API helpers for FastAPI WHO-RAG backend

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function queryRag(query: string, top_k = 5, user_id = "default_user") {
  const res = await fetch(`${API}/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k, user_id, include_audit: true }),
  });
  if (!res.ok) throw new Error(`query failed ${res.status}`);
  return res.json();
}

export async function logGlucose(payload: {
  user_id: string;
  value_mgdl: number;
  context: string;
  notes?: string;
  measured_at?: string;
}) {
  const res = await fetch(`${API}/v1/glucose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`glucose log failed ${res.status}`);
  return res.json();
}

export async function getGlucose(user_id: string, limit = 50, days?: number) {
  const url = new URL(`${API}/v1/glucose/${encodeURIComponent(user_id)}`);
  url.searchParams.set("limit", String(limit));
  if (days) url.searchParams.set("days", String(days));
  const res = await fetch(url.toString(), { cache: "no-store" });
  if (!res.ok) throw new Error(`get glucose failed ${res.status}`);
  return res.json();
}

export async function generateSoap(user_id: string, days = 14) {
  const res = await fetch(`${API}/v1/soap/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, days, language: "vi" }),
  });
  if (!res.ok) throw new Error(`soap failed ${res.status}`);
  return res.json();
}

export async function generateSoapMarkdown(user_id: string, days = 14) {
  const res = await fetch(`${API}/v1/soap/generate/markdown`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, days, language: "vi" }),
  });
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

export const API_BASE = API;
