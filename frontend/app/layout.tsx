import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";

export const metadata: Metadata = {
  title: "Diabetes RAG — WHO-RAG Assistant",
  description: "Trợ lý đái tháo đường: theo dõi đường huyết (Hướng A), tóm tắt tái khám SOAP (Hướng B), WHO-RAG API (Hướng C)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body className="min-h-screen bg-slate-50 text-slate-900">
        <Header />
        <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">{children}</main>
        <footer className="mx-auto max-w-6xl px-4 py-6 text-center text-xs text-slate-500">
          Diabetes WHO-RAG • FastAPI + Next.js • Không thay thế chỉ định bác sĩ • Built for B2B2C
        </footer>
      </body>
    </html>
  );
}
