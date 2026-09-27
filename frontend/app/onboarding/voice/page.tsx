"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import VoicePersonaPicker from "@/components/VoicePersonaPicker";

export default function VoiceOnboardingPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.push("/login");
  }, [loading, user, router]);

  if (loading) return <div className="mx-auto max-w-2xl p-6 text-sm">Đang tải…</div>;
  if (!user) return null;

  return (
    <div className="mx-auto max-w-2xl rounded-2xl border bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-700">
      <h1 className="text-lg font-bold">Chọn người đồng hành 🎧</h1>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        Mỗi giọng là một người bạn khác nhau. Bác bấm <b>Nghe thử</b> rồi chọn người mình thích nhất nhé.
      </p>
      <div className="mt-4">
        <VoicePersonaPicker userId={user.id} onDone={() => router.push("/onboarding/learn")} />
      </div>
    </div>
  );
}
