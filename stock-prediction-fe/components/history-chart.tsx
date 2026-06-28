"use client";

import React, { useEffect, useState } from "react";
import { fetchHistory, HistoricalPrediction } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from "recharts";
import { LineChart as LineChartIcon, SlidersHorizontal, CandlestickChart, Calendar } from "lucide-react";

export function HistoryChart() {
  const [allHistory, setAllHistory] = useState<HistoricalPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState<"1M" | "3M" | "6M" | "ALL">("3M");
  const [visibleModels, setVisibleModels] = useState({
    actual: true,
    hybrid_univariat: true,
    hybrid_multivariat: false,
    lstm_standalone: false,
  });

  useEffect(() => {
    async function load() {
      const data = await fetchHistory();
      setAllHistory(data);
      setLoading(false);
    }
    load();
  }, []);

  // Filter dataset based on selected timeframe (assuming ~22 trading days per month)
  const displayedHistory = React.useMemo(() => {
    if (timeframe === "1M") return allHistory.slice(-22);
    if (timeframe === "3M") return allHistory.slice(-66);
    if (timeframe === "6M") return allHistory.slice(-130);
    return allHistory;
  }, [allHistory, timeframe]);

  const toggleModel = (key: keyof typeof visibleModels) => {
    setVisibleModels((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // Show dots on tighter timeframes so individual daily points stand out sharply
  const showDots = timeframe === "1M" || timeframe === "3M";

  return (
    <Card className="border-border/60 bg-[#131722] text-[#d1d4dc] shadow-2xl w-full overflow-hidden">
      <CardHeader className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-[#2a2e39] bg-[#1c2030]/50">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30 flex items-center gap-1">
              <CandlestickChart className="h-3 w-3" /> TradingView Style
            </span>
            <span className="text-[10px] uppercase font-mono text-[#787b86]">Day-Level Granularity</span>
          </div>
          <CardTitle className="text-xl font-bold flex items-center gap-2 mt-1.5 text-white">
            <LineChartIcon className="h-5 w-5 text-orange-500" />
            IHSG Daily Closing Price Trajectory
          </CardTitle>
          <CardDescription className="text-xs sm:text-sm text-[#787b86]">
            Inspect daily market volatility. Select tighter timeframes to zoom in on individual trading days.
          </CardDescription>
        </div>
        
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          {/* Timeframe Selector */}
          <div className="flex items-center gap-1 bg-[#131722] p-1 rounded-xl border border-[#2a2e39]">
            <span className="text-[11px] font-semibold text-[#787b86] px-2 flex items-center gap-1">
              <Calendar className="h-3 w-3" /> Zoom:
            </span>
            {(["1M", "3M", "6M", "ALL"] as const).map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`text-xs px-2.5 py-1 rounded-lg font-bold transition-all ${
                  timeframe === tf
                    ? "bg-orange-600 text-white shadow-sm shadow-orange-500/20"
                    : "text-[#787b86] hover:text-white"
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Model Signals Toggle */}
          <div className="flex flex-wrap items-center gap-1.5 bg-[#131722] p-1 rounded-xl border border-[#2a2e39]">
            <span className="text-[11px] font-semibold text-[#787b86] px-2 flex items-center gap-1">
              <SlidersHorizontal className="h-3 w-3" /> Signals:
            </span>
            <button
              onClick={() => toggleModel("actual")}
              className={`text-xs px-2 py-1 rounded-lg font-medium transition-all ${
                visibleModels.actual ? "bg-[#2a2e39] text-white shadow-sm border border-[#434651]" : "text-[#787b86] hover:text-white opacity-50"
              }`}
            >
              Actual IHSG
            </button>
            <button
              onClick={() => toggleModel("hybrid_univariat")}
              className={`text-xs px-2 py-1 rounded-lg font-medium transition-all ${
                visibleModels.hybrid_univariat ? "bg-orange-600 text-white shadow-sm shadow-orange-500/20 font-bold" : "text-[#787b86] hover:text-white opacity-50"
              }`}
            >
              Hybrid Univariat (Best)
            </button>
            <button
              onClick={() => toggleModel("hybrid_multivariat")}
              className={`text-xs px-2 py-1 rounded-lg font-medium transition-all ${
                visibleModels.hybrid_multivariat ? "bg-amber-600 text-white shadow-sm shadow-amber-500/20" : "text-[#787b86] hover:text-white opacity-50"
              }`}
            >
              Hybrid Multivariat
            </button>
            <button
              onClick={() => toggleModel("lstm_standalone")}
              className={`text-xs px-2 py-1 rounded-lg font-medium transition-all ${
                visibleModels.lstm_standalone ? "bg-rose-600 text-white shadow-sm shadow-rose-500/20" : "text-[#787b86] hover:text-white opacity-50"
              }`}
            >
              LSTM Standalone
            </button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="pt-6 bg-[#131722]">
        {loading ? (
          <Skeleton className="w-full h-[420px] rounded-xl bg-[#2a2e39]" />
        ) : (
          <div className="w-full h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayedHistory} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2e39" opacity={0.8} />
                <XAxis
                  dataKey="date"
                  stroke="#787b86"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "#2a2e39" }}
                  dy={10}
                  minTickGap={15}
                />
                <YAxis
                  stroke="#787b86"
                  fontSize={11}
                  tickLine={false}
                  axisLine={{ stroke: "#2a2e39" }}
                  domain={["auto", "auto"]}
                  tickFormatter={(val) => `${val.toLocaleString()}`}
                  dx={-10}
                />
                <Tooltip
                  cursor={{ stroke: "#f97316", strokeWidth: 1.5, strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "#1c2030",
                    borderColor: "#363a45",
                    borderRadius: "8px",
                    color: "#fff",
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.8)",
                    fontSize: "12px",
                    fontFamily: "var(--font-poppins)",
                  }}
                  formatter={(value: any, name: any) => [
                    Number(value).toLocaleString("id-ID", { minimumFractionDigits: 2 }),
                    name === "actual"
                      ? "Actual IHSG"
                      : name === "hybrid_univariat"
                      ? "Hybrid Univariat"
                      : name === "hybrid_multivariat"
                      ? "Hybrid Multivariat"
                      : "LSTM Standalone",
                  ]}
                  labelFormatter={(label) => `Trading Day: ${label}`}
                />
                <Legend
                  verticalAlign="top"
                  height={36}
                  formatter={(value) => (
                    <span className="text-xs font-medium text-[#d1d4dc]">
                      {value === "actual"
                        ? "Actual IHSG"
                        : value === "hybrid_univariat"
                        ? "Hybrid Univariat (MAPE 0.83%)"
                        : value === "hybrid_multivariat"
                        ? "Hybrid Multivariat"
                        : "LSTM Standalone"}
                    </span>
                  )}
                />
                {visibleModels.actual && (
                  <Line
                    type="linear"
                    dataKey="actual"
                    stroke="#d1d4dc"
                    strokeWidth={2}
                    dot={showDots ? { r: 3, fill: "#d1d4dc" } : false}
                    activeDot={{ r: 6, fill: "#ffffff", stroke: "#131722", strokeWidth: 2 }}
                  />
                )}
                {visibleModels.hybrid_univariat && (
                  <Line
                    type="linear"
                    dataKey="hybrid_univariat"
                    stroke="#ff8a00"
                    strokeWidth={2.5}
                    dot={showDots ? { r: 3.5, fill: "#ff8a00" } : false}
                    activeDot={{ r: 7, fill: "#ff8a00", stroke: "#ffffff", strokeWidth: 2 }}
                  />
                )}
                {visibleModels.hybrid_multivariat && (
                  <Line
                    type="linear"
                    dataKey="hybrid_multivariat"
                    stroke="#fbbf24"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={showDots ? { r: 3, fill: "#fbbf24" } : false}
                  />
                )}
                {visibleModels.lstm_standalone && (
                  <Line
                    type="linear"
                    dataKey="lstm_standalone"
                    stroke="#f43f5e"
                    strokeWidth={2}
                    dot={showDots ? { r: 3, fill: "#f43f5e" } : false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
