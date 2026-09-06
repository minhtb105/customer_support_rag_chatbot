"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";

export default function RegisterPage() {
  const { register } = useAuth();
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg("");
    try {
      await register({ username, email: email || undefined, password, full_name: fullName || undefined });
      router.push("/");
    } catch (err: any) {
      setMsg(err.message);
    }
  };
  return (
    <div className="mx-auto max-w-md rounded-2xl border bg-white p-6 shadow-sm">
      <h1 className="text-lg font-bold">Đăng ký</h1>
      <p className="text-xs text-slate-500">Mặc định role <b>user</b>. Expert do admin tạo/duyệt.</p>
      <form onSubmit={submit} className="mt-4 space-y-3">
        <input placeholder="Username *" value={username} onChange={(e)=>setUsername(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" required />
        <input placeholder="Email" value={email} onChange={(e)=>setEmail(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" />
        <input placeholder="Full name" value={fullName} onChange={(e)=>setFullName(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" />
        <input type="password" placeholder="Password *" value={password} onChange={(e)=>setPassword(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" required />
        <button type="submit" className="w-full rounded-lg bg-emerald-600 py-2.5 text-sm font-medium text-white hover:bg-emerald-700">Tạo tài khoản</button>
        {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      </form>
      <div className="mt-3 text-xs"><Link href="/login" className="underline">Đã có tài khoản? Đăng nhập</Link></div>
    </div>
  );
}
