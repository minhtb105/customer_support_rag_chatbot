"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";
import { useIsDark } from "./ThemeToggle";

// Shared glucose line chart (tracker MyTracker + PrevisitView sparkline).
// Points carry a precomputed anomaly flag (server list first, local >250/<70 fallback at call sites).
export interface GlucosePoint { time: string; value: number; anomaly: boolean; }

export default function GlucoseChart({ points, yDomain, emptyHint }: {
  points: GlucosePoint[];
  yDomain?: [number, number];
  emptyHint?: string;
}) {
  const dark = useIsDark();
  const gridStroke = dark ? "#334155" : "#e2e8f0";
  const lineStroke = dark ? "#60a5fa" : "#0ea5e9";
  if (!points.length) {
    return <div className="flex h-full items-center justify-center text-sm text-slate-500">{emptyHint || "Chưa có dữ liệu"}</div>;
  }
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={points}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
        <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" minTickGap={40} stroke={gridStroke} />
        <YAxis domain={yDomain || [40, 320]} tick={{ fontSize: 10 }} stroke={gridStroke} />
        <Tooltip contentStyle={dark ? { backgroundColor: "#0f172a", borderColor: "#334155", color: "#e2e8f0" } : undefined} />
        <ReferenceLine y={126} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: "126", fontSize: 10 }} />
        <ReferenceLine y={200} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "200", fontSize: 10 }} />
        <ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 4" />
        <Line
          type="monotone" dataKey="value" stroke={lineStroke} strokeWidth={2}
          dot={(props: any) => {
            const hot = !!props?.payload?.anomaly;
            return <circle cx={props.cx} cy={props.cy} r={hot ? 5 : 3} fill={hot ? "#ef4444" : "#0ea5e9"} stroke="#fff" strokeWidth={1} />;
          }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
