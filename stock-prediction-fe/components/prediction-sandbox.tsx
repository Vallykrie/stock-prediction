"use client";

import React, { useState } from "react";
import { fetchPrediction, PredictionResult } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Search, Cpu, Zap, ArrowUpRight, ArrowDownRight, RefreshCw, Activity } from "lucide-react";

const POPULAR_TICKERS = [
  { symbol: "^JKSE", name: "IHSG (Composite)" },
  { symbol: "BBCA.JK", name: "Bank Central Asia" },
  { symbol: "BBRI.JK", name: "Bank Rakyat Indonesia" },
  { symbol: "TLKM.JK", name: "Telkom Indonesia" },
  { symbol: "AAPL", name: "Apple Inc." },
];

export function PredictionSandbox() {
  const [ticker, setTicker] = useState("^JKSE");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (symbolToPredict?: string) => {
    const target = symbolToPredict || ticker;
    if (!target) return;

    setLoading(true);
    setError(null);
    try {
      const data = await fetchPrediction(target);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to generate prediction. Please check ticker symbol.");
    } finally {
      setLoading(false);
    }
  };

  const priceDiff = result ? result.predicted_price - result.last_close : 0;
  const percentDiff = result ? (priceDiff / result.last_close) * 100 : 0;
  const isPositive = priceDiff >= 0;

  return (
    <Card className="border-border/60 bg-gradient-to-b from-card/80 to-card/40 backdrop-blur shadow-xl overflow-hidden hover:border-orange-500/30 transition-all duration-300">
      <CardHeader className="border-b border-border/40 bg-muted/20 pb-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <CardTitle className="text-xl font-bold flex items-center gap-2">
              <Activity className="h-5 w-5 text-orange-500 animate-pulse" />
              Real-Time Hybrid SARIMAX-LSTM Forecast
            </CardTitle>
            <CardDescription className="text-xs sm:text-sm">
              Query any Yahoo Finance ticker for instant next-day price prediction powered by Redis caching & live neural training.
            </CardDescription>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {POPULAR_TICKERS.map((t) => (
              <button
                key={t.symbol}
                onClick={() => {
                  setTicker(t.symbol);
                  handlePredict(t.symbol);
                }}
                className="text-xs px-2.5 py-1 rounded-full bg-secondary/80 hover:bg-orange-500/20 hover:text-orange-400 hover:border-orange-500/40 text-secondary-foreground transition-all font-medium border border-border/40"
              >
                {t.symbol}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-6 space-y-6">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handlePredict();
          }}
          className="flex gap-3"
        >
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="Enter ticker (e.g. ^JKSE, BBCA.JK, NVDA)..."
              className="pl-9 h-11 bg-background/60 border-border/60 focus-visible:ring-orange-500 font-mono font-medium"
            />
          </div>
          <Button
            type="submit"
            disabled={loading || !ticker}
            className="h-11 px-6 bg-gradient-to-r from-orange-500 to-amber-600 hover:from-orange-600 hover:to-amber-700 text-white font-semibold shadow-md transition-all duration-300 hover:shadow-orange-500/25"
          >
            {loading ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Predicting...
              </>
            ) : (
              "Generate Forecast"
            )}
          </Button>
        </form>

        {loading && (
          <div className="p-8 rounded-xl border border-dashed border-orange-500/40 flex flex-col items-center justify-center text-center space-y-3 bg-orange-500/5 animate-pulse">
            <div className="h-12 w-12 rounded-full bg-orange-500/10 flex items-center justify-center text-orange-500">
              <Cpu className="h-6 w-6 animate-spin" />
            </div>
            <div>
              <p className="font-semibold text-foreground">Processing Hybrid Pipeline...</p>
              <p className="text-xs text-muted-foreground max-w-sm mt-1">
                Checking Redis cache. If cache miss, executing SARIMAX Grid Search + fitting 30-step residual LSTM neural network (~15-30s).
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive text-sm font-medium text-center">
            ⚠️ {error}
          </div>
        )}

        {result && !loading && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-5 rounded-xl bg-gradient-to-br from-background via-muted/30 to-background border border-border/60 shadow-inner">
            <div className="space-y-1 border-b md:border-b-0 md:border-r border-border/40 pb-4 md:pb-0 md:pr-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">Target Ticker</span>
                <Badge
                  variant="outline"
                  className={
                    result.source === "cache"
                      ? "bg-amber-500/10 text-amber-500 dark:text-amber-400 border-amber-500/30 gap-1 text-[10px]"
                      : result.source === "model"
                      ? "bg-orange-500/10 text-orange-500 dark:text-orange-400 border-orange-500/30 gap-1 text-[10px]"
                      : "bg-slate-500/10 text-slate-500 border-slate-500/30 gap-1 text-[10px]"
                  }
                >
                  {result.source === "cache" ? (
                    <>
                      <Zap className="h-3 w-3 fill-current" /> Redis Cache HIT
                    </>
                  ) : result.source === "model" ? (
                    <>
                      <Cpu className="h-3 w-3" /> Live Trained
                    </>
                  ) : (
                    "Fallback Preview"
                  )}
                </Badge>
              </div>
              <p className="text-3xl font-extrabold tracking-tight font-mono text-foreground">{result.ticker}</p>
              <p className="text-xs text-muted-foreground">Forecast Date: <span className="font-medium text-foreground">{result.prediction_date}</span></p>
            </div>

            <div className="space-y-1 border-b md:border-b-0 md:border-r border-border/40 pb-4 md:pb-0 md:pr-4">
              <span className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">Predicted Close</span>
              <div className="flex items-baseline gap-2">
                <p className="text-3xl font-extrabold tracking-tight font-mono text-foreground">
                  {result.predicted_price.toLocaleString()}
                </p>
              </div>
              <p className="text-xs text-muted-foreground">Last Close ({result.last_close_date}): <span className="font-mono">{result.last_close.toLocaleString()}</span></p>
            </div>

            <div className="space-y-1 flex flex-col justify-center">
              <span className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">Expected Trajectory</span>
              <div className="flex items-center gap-2">
                <span
                  className={`flex items-center gap-1 text-2xl font-bold font-mono px-2.5 py-1 rounded-lg ${
                    isPositive
                      ? "bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border border-emerald-500/20"
                      : "bg-rose-500/10 text-rose-500 dark:text-rose-400 border border-rose-500/20"
                  }`}
                >
                  {isPositive ? <ArrowUpRight className="h-6 w-6 stroke-[2.5]" /> : <ArrowDownRight className="h-6 w-6 stroke-[2.5]" />}
                  {isPositive ? "+" : ""}
                  {priceDiff.toFixed(2)} ({percentDiff.toFixed(2)}%)
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground mt-1">Rolling one-step ahead estimate.</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
