"use client";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import PrevisitView from "@/components/PrevisitView";

export default function DoctorPatientDetail({ params }: { params: { id: string } }){
  const { user, loading, isExpert, isAdmin } = useAuth();
  if(loading) return <div className="rounded-2xl border bg-white p-8 text-center text-sm text-slate-500">Đang tải...</div>;
  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center">Cần đăng nhập <Link href="/login" className="underline">Đăng nhập</Link></div>;
  if(!(isExpert || isAdmin)) return <div className="rounded-2xl border bg-white p-8 text-center text-sm text-red-700">403 — Chỉ bác sĩ mới xem được hồ sơ bệnh nhân.</div>;
  return (
    <div className="space-y-4">
      <Link href="/expert/patients" className="text-xs text-blue-700 underline">← Danh sách bệnh nhân</Link>
      <PrevisitView patientId={params.id} title={`Hồ sơ bệnh nhân — ${params.id.slice(0,8)}`} subtitle="Smart queue → pre-visit: sparkline 90 ngày + SOAP, P để trống cho bác sĩ chỉ định." />
    </div>
  );
}
