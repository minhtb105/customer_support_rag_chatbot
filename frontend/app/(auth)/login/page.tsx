"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

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
    <Card className="mx-auto max-w-md p-6">
      <h1 className="text-lg font-bold">Đăng nhập</h1>
      <p className="text-xs text-slate-500 dark:text-slate-400">Bắt buộc để dùng WHO-RAG. Cookie httpOnly an toàn.</p>
      <form onSubmit={submit} className="mt-4 space-y-3">
        <Input placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} required />
        <Input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        <Button type="submit" variant="dark" className="w-full">Đăng nhập</Button>
        {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      </form>
      <div className="mt-3 text-xs text-slate-600 dark:text-slate-300">Chưa có tài khoản? <Link href="/register" className="underline">Đăng ký</Link></div>
      <div className="mt-3 text-xs text-slate-500 dark:text-slate-400">Admin cần tạo expert: đăng nhập admin → Quản trị → Tạo user role doctor/pharmacist/specialist.</div>
    </Card>
  );
}
