"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

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
      router.push("/onboarding/voice");
    } catch (err: any) {
      setMsg(err.message);
    }
  };
  return (
    <Card className="mx-auto max-w-md p-6">
      <h1 className="text-lg font-bold">Đăng ký</h1>
      <p className="text-xs text-slate-500 dark:text-slate-400">Mặc định role <b>user</b>. Expert do admin tạo/duyệt.</p>
      <form onSubmit={submit} className="mt-4 space-y-3">
        <Input placeholder="Username *" value={username} onChange={(e)=>setUsername(e.target.value)} required />
        <Input placeholder="Email" value={email} onChange={(e)=>setEmail(e.target.value)} />
        <Input placeholder="Full name" value={fullName} onChange={(e)=>setFullName(e.target.value)} />
        <Input type="password" placeholder="Password *" value={password} onChange={(e)=>setPassword(e.target.value)} required />
        <Button type="submit" variant="success" className="w-full">Tạo tài khoản</Button>
        {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      </form>
      <div className="mt-3 text-xs"><Link href="/login" className="underline">Đã có tài khoản? Đăng nhập</Link></div>
    </Card>
  );
}
