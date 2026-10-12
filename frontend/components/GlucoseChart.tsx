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
  const lineStroke = "var(--med-primary)";
  const anomalyStroke = "var(--med-critical)";
  const warnStroke = "var(--med-warning)";
  if (!points.length) {
    return <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">{emptyHint || "Chưa có dữ liệu"}</div>;
  }
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={points}>
        <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
        <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" minTickGap={40} stroke={gridStroke} />
        <YAxis domain={yDomain || [40, 320]} tick={{ fontSize: 10 }} stroke={gridStroke} />
        <Tooltip contentStyle={dark ? { backgroundColor: "#0f172a", borderColor: "#334155", color: "#e2e8f0" } : undefined} />
        <ReferenceLine y={126} stroke={warnStroke} strokeDasharray="4 4" label={{ value: "126", fontSize: 10 }} />
        <ReferenceLine y={200} stroke={anomalyStroke} strokeDasharray="4 4" label={{ value: "200", fontSize: 10 }} />
        <ReferenceLine y={70} stroke={warnStroke} strokeDasharray="4 4" />
        <Line
          type="monotone" dataKey="value" stroke={lineStroke} strokeWidth={2}
          dot={(props: any) => {
            const hot = !!props?.payload?.anomaly;
            if (hot) {
              // distinct shape (triangle), not color-only, for colorblind users
              const { cx, cy } = props;
              return <path d={`M ${cx} ${cy - 6} L ${cx + 5.5} ${cy + 4} L ${cx - 5.5} ${cy + 4} Z`} fill={anomalyStroke} stroke="#fff" strokeWidth={1} />;
            }
            return <circle cx={props.cx} cy={props.cy} r={3} fill={lineStroke} stroke="#fff" strokeWidth={1} />;
          }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
