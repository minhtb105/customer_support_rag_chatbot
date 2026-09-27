"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { getLessons, getTodayLesson } from "@/lib/api";

const GROUP_LABEL: Record<string, string> = {
  "hieu-benh": "Hiểu bệnh",
  "dinh-duong": "Dinh dưỡng",
  "van-dong": "Vận động",
  "thuoc-theo-doi": "Thuốc và theo dõi",
  "bien-chung": "Biến chứng",
  "tam-ly": "Tâm lý",
};

export default function HocPage() {
  const { user, loading } = useAuth();
  const [data, setData] = useState<any>(null);
  const [today, setToday] = useState<any>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (loading || !user) return;
    (async () => {
      try {
        setData(await getLessons(user.id));
        setToday(await getTodayLesson(user.id));
      } catch (e: any) {
        setErr(e.message);
      }
    })();
  }, [loading, user]);

  if (loading) return <div className="mx-auto max-w-4xl p-6 text-sm">Đang tải…</div>;
  if (!user) return <div className="mx-auto max-w-4xl p-6 text-sm">Bác đăng nhập để học nhé.</div>;

  const lessons: any[] = data?.lessons || [];
  const doneCount = lessons.filter((l) => l.status === "done").length;

  return (
    <div className="mx-auto max-w-4xl p-6">
      <h1 className="text-lg font-bold">📚 Học mỗi ngày một chút</h1>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        Đã học {doneCount}/{lessons.length} bài. Mỗi bài nghe khoảng 1 phút.
      </p>
      {today?.lesson && (
        <div className="mt-4 rounded-2xl border-2 border-emerald-500 bg-emerald-50 p-4 dark:bg-emerald-950 dark:border-emerald-700">
          <div className="text-sm font-bold">☀️ Hôm nay (ngày {today.day}/7): {today.lesson.title}</div>
          <Link href={`/hoc/${today.lesson.id}`} className="mt-2 inline-block rounded-lg bg-emerald-600 px-6 py-3 text-sm font-bold text-white min-h-[44px]">
            Học ngay
          </Link>
        </div>
      )}
      {err && <div className="mt-3 text-sm text-red-600">{err}</div>}
      <div className="mt-4 space-y-4">
        {Object.entries(GROUP_LABEL).map(([g, label]) => {
          const items = lessons.filter((l) => l.group === g);
          if (!items.length) return null;
          return (
            <div key={g} className="rounded-2xl border bg-white p-4 shadow-sm dark:bg-slate-900 dark:border-slate-700">
              <h2 className="text-sm font-bold">{label} ({items.filter((l) => l.status === "done").length}/{items.length})</h2>
              <ul className="mt-2 space-y-1">
                {items.map((l) => (
                  <li key={l.id}>
                    <Link href={`/hoc/${l.id}`} className="block rounded-lg px-2 py-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-800 min-h-[44px]">
                      {l.status === "done" ? "✅ " : "⬜ "}{l.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}
