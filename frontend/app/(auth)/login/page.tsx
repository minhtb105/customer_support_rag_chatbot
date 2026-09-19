"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg("");
    try {
      await login(username, password);
      router.push("/");
    } catch (err: any) {
      setMsg(err.message);
    }
  };

  return (
    <div className="mx-auto max-w-md rounded-2xl border bg-white p-6 shadow-sm">
      <h1 className="text-lg font-bold">Đăng nhập</h1>
      <p className="text-xs text-slate-500">Bắt buộc để dùng WHO-RAG. Cookie httpOnly an toàn.</p>
      <form onSubmit={submit} className="mt-4 space-y-3">
        <input placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" required />
        <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" required />
        <button type="submit" className="w-full rounded-lg bg-slate-900 py-2.5 text-sm font-medium text-white hover:bg-black">Đăng nhập</button>
        {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      </form>
      <div className="mt-3 text-xs text-slate-600">Chưa có tài khoản? <Link href="/register" className="underline">Đăng ký</Link></div>
      <div className="mt-3 text-xs text-slate-500">Admin cần tạo expert: đăng nhập admin → Quản trị → Tạo user role doctor/pharmacist/specialist.</div>
    </div>
  );
}
