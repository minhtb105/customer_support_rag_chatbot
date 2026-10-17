"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { getLessons, getTodayLesson } from "@/lib/api";
import { BookOpen, Sun, CheckCircle2, Circle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";

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
      <h1 className="text-lg font-bold flex items-center gap-2"><BookOpen size={18} aria-hidden /> Học mỗi ngày một chút</h1>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        Đã học {doneCount}/{lessons.length} bài. Mỗi bài nghe khoảng 1 phút.
      </p>
      {today?.lesson && (
        <div className="mt-4 rounded-2xl border-2 border-emerald-500 bg-emerald-50 p-4 dark:bg-emerald-950 dark:border-emerald-700">
          <div className="text-sm font-bold flex items-center gap-1.5"><Sun size={15} aria-hidden /> Hôm nay (ngày {today.day}/7): {today.lesson.title}</div>
          <Link href={`/hoc/${today.lesson.id}`} className={buttonVariants({ variant: "success", className: "mt-2 font-bold" })}>
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
            <Card key={g} className="p-4">
              <h2 className="text-sm font-bold">{label} ({items.filter((l) => l.status === "done").length}/{items.length})</h2>
              <ul className="mt-2 space-y-1">
                {items.map((l) => (
                  <li key={l.id}>
                    <Link href={`/hoc/${l.id}`} className="flex items-center gap-1.5 rounded-lg px-2 py-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-800 min-h-[44px]">
                      {l.status === "done" ? <CheckCircle2 size={14} aria-hidden className="text-emerald-600" /> : <Circle size={14} aria-hidden className="text-slate-400" />}{l.title}
                    </Link>
                  </li>
                ))}
              </ul>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
