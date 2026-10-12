"use client";
import { useEffect, useState } from "react";

// Shared fetch-list shell for monitor pages (guidelines/alerts).
// Per-page filters + item render + decide/trigger stay local (different
// contracts); only the loading/msg/total/list boilerplate is shared.
export function useMonitorList<T>(loader: () => Promise<{ items: T[]; total: number }>, deps: any[] = []) {
  const [items, setItems] = useState<T[]>([]);
  const [total, setTotal] = useState(0);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const fetchList = async () => {
    setLoading(true);
    try {
      const j = await loader();
      setItems(j.items || []);
      setTotal(j.total || 0);
    } catch (e: any) {
      setMsg(e.message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { fetchList(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, deps);
  return { items, total, msg, setMsg, loading, fetchList };
}

export function MonitorMsg({ msg }: { msg: string }) {
  if (!msg) return null;
  return <div className="rounded-lg border bg-slate-50 p-3 text-xs dark:bg-slate-800 dark:border-slate-700 dark:text-slate-200">{msg}</div>;
}

export function MonitorLoading({ loading, empty, emptyText, children }: {
  loading: boolean; empty: boolean; emptyText: string; children: React.ReactNode;
}) {
  if (loading) return <div className="text-sm text-slate-500 dark:text-slate-400">Đang tải…</div>;
  if (empty) {
    return (
      <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm text-slate-500 dark:text-slate-400">
        {emptyText}
      </div>
    );
  }
  return <>{children}</>;
}
