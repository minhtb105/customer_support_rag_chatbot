"use client";
import { useState } from "react";
import { lookupNutrition } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

// P3 UI hookup: tra cứu món ăn -> số liệu + khẩu phần mild/severe, không LLM.
export default function NutritionLookup() {
  const [q, setQ] = useState("");
  const [profile, setProfile] = useState<"mild" | "severe" | "both">("both");
  const [res, setRes] = useState<any>(null);
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const go = async () => {
    if (!q.trim()) return;
    setLoading(true); setMsg(""); setRes(null);
    try {
      setRes(await lookupNutrition(q.trim(), profile, 3));
    } catch (e: any) { setMsg(e.message); } finally { setLoading(false); }
  };
  const rec = (m: any) =>
    profile === "severe" ? m.recommendation?.severe : m.recommendation?.mild;
  return (
    <Card className="p-5">
      <h2 className="text-sm font-semibold">Tra cứu món ăn (số liệu chuẩn)</h2>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">Hỏi “ăn bao nhiêu ổi” → số gram + GL + carb-choice, không qua AI diễn giải.</p>
      <div className="mt-3 flex flex-col gap-2 sm:flex-row">
        <Input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") go(); }} placeholder="vd: ổi, sầu riêng, trà sữa" className="flex-1" />
        <select value={profile} onChange={(e) => setProfile(e.target.value as any)} className="rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]">
          <option value="both">Cả 2 mức</option>
          <option value="mild">Kiểm soát tốt</option>
          <option value="severe">Kiểm soát kém</option>
        </select>
        <Button onClick={go} disabled={loading} variant="success" className="px-5">{loading ? "Đang tra..." : "Tra cứu"}</Button>
      </div>
      {msg && <div className="mt-2 text-xs text-red-600">{msg}</div>}
      {res && res.count === 0 && <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">Không tìm thấy món “{res.query}”. Thử tên khác (vd: chuối, phở bò, nước dừa).</div>}
      {(res?.matches || []).map((m: any) => {
        const r = rec(m) || {};
        return (
          <div key={m.food_id} className="mt-3 rounded-xl border border-emerald-100 bg-emerald-50/50 p-3 dark:bg-emerald-950/30 dark:border-emerald-900">
            <div className="text-sm font-bold">{m.vi} <span className="font-normal text-slate-500 dark:text-slate-400">({m.en})</span></div>
            <div className="mt-1 text-xs text-slate-600 dark:text-slate-300">
              Khẩu phần chuẩn {m.serving.size_g}g ({m.serving.description}) · {m.per_100g.energy_kcal} kcal/100g ·
              carbs {m.per_100g.total_carbohydrate_g}g (đường {m.per_100g.sugars_g}g, xơ {m.per_100g.dietary_fiber_g}g)
              {m.gi != null ? ` · GI ${m.gi} (${m.gi_tier})` : " · chưa có GI"}
            </div>
            <div className="mt-1 text-xs font-medium text-emerald-800 dark:text-emerald-200">
              Nên ăn: {r.serving_g}g (~{r.carb_choices} carb-choice{m.gl != null ? `, GL ${r.gl}` : ""})
              {profile === "both" && m.recommendation?.severe && m.recommendation.severe.serving_g !== r.serving_g && (
                <span> · Kiểm soát kém: {m.recommendation.severe.serving_g}g (~{m.recommendation.severe.carb_choices} carb-choice{m.recommendation.severe.gl != null ? `, GL ${m.recommendation.severe.gl}` : ""})</span>
              )}
            </div>
            <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">{m.notes}</div>
          </div>
        );
      })}
    </Card>
  );
}
