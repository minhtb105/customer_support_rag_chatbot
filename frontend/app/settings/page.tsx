"use client";
import { useAuth } from "@/lib/auth";
import VoicePersonaPicker from "@/components/VoicePersonaPicker";
import Link from "next/link";

export default function SettingsPage() {
  const { user, loading } = useAuth();

  if (loading) return <div className="mx-auto max-w-2xl p-6 text-sm">Đang tải…</div>;
  if (!user) {
    return (
      <div className="mx-auto max-w-2xl p-6 text-sm">
        Bác cần <Link href="/login" className="underline">đăng nhập</Link> để đổi giọng đọc nhé.
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl rounded-2xl border bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-700">
      <h1 className="text-lg font-bold">Cài đặt</h1>
      <h2 className="mt-4 text-sm font-semibold">🎧 Giọng người đồng hành</h2>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
        Đổi giọng đọc bất cứ lúc nào — giọng mới dùng ngay cho mọi câu trả lời.
      </p>
      <div className="mt-3">
        <VoicePersonaPicker userId={user.id} showSkip={false} />
      </div>
    </div>
  );
}
