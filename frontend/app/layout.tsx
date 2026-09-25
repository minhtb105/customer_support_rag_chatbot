import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";
import { AuthProvider } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Diabetes RAG — WHO-RAG Assistant",
  description: "Trợ lý đái tháo đường: theo dõi đường huyết (Hướng A), tóm tắt tái khám SOAP (Hướng B), WHO-RAG API (Hướng C)",
};

const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem('med-theme');if(!t){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}if(t==='dark'){document.documentElement.classList.add('dark');}}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="min-h-screen bg-slate-50 text-sm text-slate-900 dark:bg-slate-950 dark:text-slate-100">
        <AuthProvider>
          <Header />
          <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">{children}</main>
          <footer className="mx-auto max-w-6xl px-4 py-6 text-center text-xs text-slate-500 dark:text-slate-400">
            Diabetes WHO-RAG • FastAPI + Next.js • Không thay thế chỉ định bác sĩ • Built for B2B2C
          </footer>
        </AuthProvider>
      </body>
    </html>
  );
}
