"use client";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import PrevisitView from "@/components/PrevisitView";

export default function PrevisitPage(){
  const { user } = useAuth();
  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center">Cần đăng nhập <Link href="/login" className="underline">Đăng nhập</Link></div>;
  return <PrevisitView patientId={user.id} title="Hướng B — Hồ sơ trước tái khám (SOAP)" subtitle={`User ${user.username} · B2B2C cho phòng khám.`} />;
}
