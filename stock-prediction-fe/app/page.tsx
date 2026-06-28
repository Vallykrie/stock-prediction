"use client";

import React from "react";
import { PredictionSandbox } from "@/components/prediction-sandbox";
import { MetricsShowcase } from "@/components/metrics-showcase";
import { HistoryChart } from "@/components/history-chart";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Terminal, Activity, BarChart3, TrendingUp, ExternalLink, GraduationCap } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-orange-950/25 via-background to-background">
      {/* Top Glassmorphic Navigation */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex h-16 items-center justify-end px-4 sm:px-6 lg:px-8">

          <div className="flex items-center gap-4">
            <Badge variant="outline" className="hidden sm:flex items-center gap-1.5 bg-amber-500/10 text-amber-500 border-amber-500/30 text-xs px-3 py-1 font-semibold">
              <span className="h-2 w-2 rounded-full bg-amber-500 animate-ping" />
              FastAPI Online
            </Badge>
            <a
              href="https://github.com/Vallykrie/stock-prediction"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-secondary hover:bg-secondary/80 text-secondary-foreground transition-all border border-border/40 hover:border-orange-500/40"
            >
              <ExternalLink className="h-4 w-4 text-orange-400" />
              <span>Repository</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
        {/* Hero Banner */}
        <section className="text-center space-y-4 max-w-3xl mx-auto pt-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-orange-500/10 border border-orange-500/30 text-orange-400 text-xs font-semibold uppercase tracking-wider animate-fade-in">
            <GraduationCap className="h-4 w-4" /> Proposal Skripsi Teknik Informatika UB
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-foreground leading-[1.1]">
            Prediksi IHSG dengan Model{" "}
            <span className="bg-gradient-to-r from-orange-400 via-amber-400 to-yellow-400 bg-clip-text text-transparent">
              Hybrid SARIMAX-LSTM
            </span>
          </h1>
          <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto font-normal leading-relaxed">
            Menggabungkan keunggulan komponen linear <strong className="text-foreground">SARIMAX</strong> dengan kapabilitas deep learning <strong className="text-foreground">LSTM</strong> untuk memprediksi harga penutupan harian pasar saham secara akurat.
          </p>
          <div className="pt-2 flex flex-wrap items-center justify-center gap-3 text-xs text-muted-foreground font-mono">
            <span>Oleh: Pande Kadek Nathan P. S. P. (235150207111051)</span>
            <span>•</span>
            <span>Rolling One-Step-Ahead</span>
          </div>
        </section>

        {/* Dashboard Tabs */}
        <Tabs defaultValue="sandbox" className="w-full space-y-8">
          <div className="flex justify-center">
            <TabsList className="bg-muted/60 p-1.5 rounded-2xl border border-border/50 shadow-inner">
              <TabsTrigger
                value="sandbox"
                className="rounded-xl px-6 py-2.5 text-sm font-semibold transition-all data-[state=active]:bg-orange-600 data-[state=active]:text-white data-[state=active]:shadow-md data-[state=active]:shadow-orange-500/20 flex items-center gap-2"
              >
                <Activity className="h-4 w-4 text-orange-400 data-[state=active]:text-white" />
                Live Prediction Sandbox
              </TabsTrigger>
              <TabsTrigger
                value="benchmark"
                className="rounded-xl px-6 py-2.5 text-sm font-semibold transition-all data-[state=active]:bg-orange-600 data-[state=active]:text-white data-[state=active]:shadow-md data-[state=active]:shadow-orange-500/20 flex items-center gap-2"
              >
                <BarChart3 className="h-4 w-4 text-amber-400 data-[state=active]:text-white" />
                Model Benchmarks
              </TabsTrigger>
              <TabsTrigger
                value="history"
                className="rounded-xl px-6 py-2.5 text-sm font-semibold transition-all data-[state=active]:bg-orange-600 data-[state=active]:text-white data-[state=active]:shadow-md data-[state=active]:shadow-orange-500/20 flex items-center gap-2"
              >
                <TrendingUp className="h-4 w-4 text-yellow-400 data-[state=active]:text-white" />
                TradingView Trajectory
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="sandbox" className="space-y-8 animate-in fade-in-50 duration-500">
            <PredictionSandbox />
            <div className="pt-4">
              <MetricsShowcase />
            </div>
          </TabsContent>

          <TabsContent value="benchmark" className="space-y-8 animate-in fade-in-50 duration-500">
            <MetricsShowcase />
            <HistoryChart />
          </TabsContent>

          <TabsContent value="history" className="space-y-8 animate-in fade-in-50 duration-500">
            <HistoryChart />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-6 rounded-2xl border border-border/50 bg-card/40 backdrop-blur space-y-3 hover:border-orange-500/30 transition-colors">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-amber-400" /> Temuan Utama 1: Unggul Signifikan
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Model Hybrid mengalahkan LSTM Tunggal secara signifikan. Pendekatan rolling one-step-ahead pada SARIMAX, dikombinasikan dengan LSTM untuk memodelkan residual non-linear, terbukti jauh lebih akurat dibandingkan LSTM standalone (MAPE 0.83% vs 3.80%).
                </p>
              </div>
              <div className="p-6 rounded-2xl border border-border/50 bg-card/40 backdrop-blur space-y-3 hover:border-orange-500/30 transition-colors">
                <h3 className="text-lg font-bold flex items-center gap-2">
                  <Terminal className="h-5 w-5 text-orange-400" /> Temuan Utama 2: Efficient Market Hypothesis
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Model Univariat sedikit lebih unggul dari Multivariat (MAPE 0.83% vs 0.88%). Hal ini mengindikasikan bahwa harga penutupan IHSG harian sudah merangkum informasi makroekonomi, sehingga penambahan variabel eksogen mengintroduksi sedikit noise tambahan.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/40 bg-muted/10 py-8 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-muted-foreground">
          <p>© 2026 Universitas Brawijaya • Fakultas Ilmu Komputer</p>
          <p className="font-mono">Built with Next.js 16 • Tailwind CSS v4 • FastAPI • Recharts • Poppins Font</p>
        </div>
      </footer>
    </div>
  );
}
