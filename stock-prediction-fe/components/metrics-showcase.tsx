"use client";

import React, { useEffect, useState } from "react";
import { fetchMetrics, ModelMetric } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Trophy, TrendingUp, AlertCircle, Sparkles } from "lucide-react";

export function MetricsShowcase() {
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const data = await fetchMetrics();
      setMetrics(data);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
        {[1, 2, 3].map((i) => (
          <Card key={i} className="border-border/40 bg-card/60 backdrop-blur">
            <CardHeader className="space-y-2">
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-6 w-3/4" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-16 w-full" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  // Find winner (lowest MAPE)
  const winner = metrics.reduce((prev, curr) => (prev.mape < curr.mape ? prev : curr), metrics[0]);

  return (
    <div className="space-y-4 w-full">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <TrendingUp className="h-6 w-6 text-orange-500" />
            Model Benchmark Performance
          </h2>
          <p className="text-sm text-muted-foreground">
            Out-of-sample evaluation metrics using Rolling One-Step-Ahead predictions on IHSG test data.
          </p>
        </div>
        <Badge variant="outline" className="w-fit bg-orange-500/10 text-orange-400 border-orange-500/30 px-3 py-1 text-xs font-semibold">
          Rolling One-Step-Ahead
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {metrics.map((m) => {
          const isWinner = m.model === winner.model;
          return (
            <Card
              key={m.model}
              className={`relative overflow-hidden transition-all duration-300 hover:scale-[1.02] ${
                isWinner
                  ? "border-orange-500/50 bg-gradient-to-br from-orange-500/10 via-background to-background shadow-lg shadow-orange-500/10 ring-1 ring-orange-500/20"
                  : "border-border/50 bg-card/40 backdrop-blur hover:border-border"
              }`}
            >
              {isWinner && (
                <div className="absolute top-0 right-0 bg-gradient-to-l from-orange-500 to-amber-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-lg shadow-md flex items-center gap-1">
                  <Sparkles className="h-3 w-3" /> BEST MODEL
                </div>
              )}
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  {isWinner ? (
                    <Trophy className="h-5 w-5 text-orange-500 shrink-0" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-muted-foreground shrink-0" />
                  )}
                  <CardTitle className="text-base font-semibold leading-tight">
                    {m.model}
                  </CardTitle>
                </div>
                <CardDescription className="text-xs">
                  {m.model.includes("Univariat")
                    ? "Pure historical price sequence (Efficient Market Hypothesis)"
                    : m.model.includes("Multivariat")
                    ? "Includes USD/IDR, Gold (XAU), and S&P 500 exogens"
                    : "Standalone Deep Learning benchmark without SARIMAX"}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-2">
                <div className="grid grid-cols-3 gap-2 p-3 rounded-lg bg-muted/40 border border-border/30 text-center">
                  <div>
                    <p className="text-[10px] uppercase font-semibold text-muted-foreground">MAPE</p>
                    <p className={`text-lg font-bold ${isWinner ? "text-orange-500 dark:text-orange-400" : "text-foreground"}`}>
                      {m.mape.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase font-semibold text-muted-foreground">RMSE</p>
                    <p className="text-lg font-bold text-foreground">{m.rmse.toFixed(1)}</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase font-semibold text-muted-foreground">MAE</p>
                    <p className="text-lg font-bold text-foreground">{m.mae.toFixed(1)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
