"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { saveLearnOnboarding } from "@/lib/api";

export default function LearnOnboardingPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [age, setAge] = useState("");
  const [meds, setMeds] = useState("");
  const [hasMeter, setHasMeter] = useState(1);

  useEffect(() => {
    if (!loading && !user) router.push("/login");
  }, [loading, user, router]);

  if (loading) return <div className="mx-auto max-w-2xl p-6 text-sm">Đang tải…</div>;
  if (!user) return null;

  const save = async (skip: boolean) => {
    try {
      if (!skip) {
        const n = parseInt(age, 10);
        await saveLearnOnboarding(user.id, {
          ...(age.trim() && !Number.isNaN(n) ? { age: n } : {}),
          ...(meds.trim() ? { meds: meds.trim() } : {}),
          has_meter: hasMeter,
        });
      } else {
        await saveLearnOnboarding(user.id, {});
      }
    } catch { /* fail-open: still continue */ }
    router.push("/");
  };

  return (
    <div className="mx-auto max-w-2xl rounded-2xl border bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-700">
      <h1 className="text-lg font-bold">📝 Làm quen một chút</h1>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        Để con chọn bài học hợp với bác. Bác có thể bỏ qua, trả lời sau cũng được.
      </p>
      <label className="mt-4 block text-sm font-semibold">Bác năm nay bao nhiêu tuổi?</label>
      <input value={age} onChange={(e) => setAge(e.target.value)} inputMode="numeric" placeholder="Ví dụ: 65"
        className="mt-1 w-full rounded-lg border px-3 py-3 text-sm min-h-[44px] dark:bg-slate-800 dark:border-slate-600" />
      <label className="mt-4 block text-sm font-semibold">Bác đang uống thuốc gì? (viết tên nếu nhớ)</label>
      <input value={meds} onChange={(e) => setMeds(e.target.value)} placeholder="Ví dụ: thuốc tiểu đường buổi sáng"
        className="mt-1 w-full rounded-lg border px-3 py-3 text-sm min-h-[44px] dark:bg-slate-800 dark:border-slate-600" />
      <label className="mt-4 block text-sm font-semibold">Nhà mình có máy đo đường huyết không?</label>
      <div className="mt-1 flex gap-2">
        <button onClick={() => setHasMeter(1)} className={`rounded-lg px-6 py-3 text-sm font-bold min-h-[44px] ${hasMeter ? "bg-emerald-600 text-white" : "border"}`}>Có</button>
        <button onClick={() => setHasMeter(0)} className={`rounded-lg px-6 py-3 text-sm font-bold min-h-[44px] ${!hasMeter ? "bg-emerald-600 text-white" : "border"}`}>Chưa có</button>
      </div>
      <div className="mt-5 flex gap-2">
        <button onClick={() => save(false)} className="rounded-lg bg-blue-600 px-6 py-3 text-sm font-bold text-white min-h-[44px]">Xong</button>
        <button onClick={() => save(true)} className="rounded-lg border px-6 py-3 text-sm min-h-[44px]">Bỏ qua</button>
      </div>
    </div>
  );
}
