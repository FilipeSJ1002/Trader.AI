# -*- coding: utf-8 -*-
"""
v6_exp_atr.py — Trader.AI V6: stops adaptativos por volatilidade (ATR)
=======================================================================

HIPOTESE (com causa medida):
  O SL fixo de 0,5% e MENOR que o ruido natural de ativos volateis.
     ADA  vol 2h = 1,08%  -> PnL -233  (win 14%)  estopa por ruido
     DOT  vol 2h = 1,11%  -> PnL -163  (win 21%)
     LTC  vol 2h = 0,81%  -> PnL  +88  (win 47%)  ruido cabe no stop
     BTC  vol 2h = 0,68%  -> PnL  +12
  Escalando o stop pelo ATR do PROPRIO ativo, o stop fica sempre fora do
  ruido dele — solucao estrutural, sem escolher ativos a dedo.

CALIBRACAO DO k (medida, nao chutada):
  atr_pct e o ATR(14) em escala de 1 MINUTO: 0,064% (BTC) a 0,121% (ADA/DOT).
  Portanto SL = k x atr_pct precisa de k na casa das unidades, NAO 1.2:
     k=6 -> BTC 0,39% | LTC 0,53% | ADA 0,72% | DOT 0,73%
     k=7 -> BTC 0,45% | LTC 0,62% | ADA 0,85% | DOT 0,85%
     k=8 -> BTC 0,51% | LTC 0,71% | ADA 0,97% | DOT 0,97%
  (com SL fixo de 0,5%, BTC opera a k~7,8 e ADA a k~4,1 — e por isso que os
   volateis sofrem: o stop deles e RELATIVAMENTE muito mais apertado.)

Uso: python v6_exp_atr.py
"""
import os
import re
import sys
import subprocess
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
REL  = os.path.join(BASE, "relatorios")   # saidas ficam concentradas aqui (gitignored)
os.makedirs(REL, exist_ok=True)
PY   = sys.executable
LOG  = os.path.join(REL, "v6_exp_atr.log")

MODELO  = "v5_model_b.pth"
HOLDOUT = ("2026-06-01", "2026-07-24")
ALL     = ["--assets", "ALL"]

# (nome, flags, descricao)
CENARIOS = [
    ("11_fixo",  ALL,                                   "11 ativos · SL fixo 0,5% (baseline ruim)"),
    ("11_k6",    ALL + ["--sl-mode", "atr", "--atr-k", "6"],  "11 ativos · SL 6xATR"),
    ("11_k7",    ALL + ["--sl-mode", "atr", "--atr-k", "7"],  "11 ativos · SL 7xATR"),
    ("11_k8",    ALL + ["--sl-mode", "atr", "--atr-k", "8"],  "11 ativos · SL 8xATR"),
    ("6_k7",     ["--sl-mode", "atr", "--atr-k", "7"],        "6 ativos · SL 7xATR"),
]

PERIODOS = [("teste", [], "teste jan-jul/26"),
            ("holdout", ["--from", HOLDOUT[0], "--to", HOLDOUT[1]], "holdout jun-jul/26")]

# Referencias ja medidas com SL fixo (para a tabela final)
BASELINE = {("6_fixo", "teste"):   {"ret": +2.0, "ops": 68},
            ("6_fixo", "holdout"): {"ret": +0.2, "ops": 17}}


def log(msg=""):
    line = f"[{datetime.now():%H:%M:%S}] {msg}" if msg else ""
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def metricas(path):
    if not os.path.exists(path):
        return None
    t = open(path, encoding="utf-8", errors="replace").read()
    def g(pat, cast=float, d=None):
        m = re.search(pat, t)
        return cast(m.group(1).replace(",", "")) if m else d
    return {
        "ret":    g(r"Capital final.*?\(([+-][\d.]+)%\)"),
        "ops":    g(r"Total de operacoes\s*:\s*(\d+)", int),
        "winpct": g(r"Vitorias\s*:\s*\d+\s*\(([\d.]+)%\)"),
        "liq":    g(r"Liquidacoes\s*:\s*(\d+)", int),
        "sl":     g(r"SL\s+:\s*(\d+)\s*ops", int, 0),
        "tp":     g(r"TP\s+:\s*(\d+)\s*ops", int, 0),
    }


def main():
    open(LOG, "w", encoding="utf-8").close()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""     # CPU — paralelo ao treino na GPU

    log("=" * 78)
    log("V6 — STOPS ADAPTATIVOS (k calibrado na escala real do atr_pct)")
    log("Baseline conhecido: 6 ativos SL fixo = +2,0% (teste) | +0,2% (holdout)")
    log("=" * 78)

    res = {}
    for nome, flags, desc in CENARIOS:
        for pkey, pflags, pdesc in PERIODOS:
            out = f"v6_atr_{nome}_{pkey}.txt"
            log(f"[RODANDO] {desc} · {pdesc}")
            with open(os.path.join(REL, out), "w", encoding="utf-8") as fh:
                rc = subprocess.run([PY, "v5_backtest.py", "--model", MODELO]
                                    + flags + pflags,
                                    cwd=BASE, env=env, stdout=fh,
                                    stderr=subprocess.STDOUT).returncode
            m = metricas(os.path.join(REL, out))
            res[(nome, pkey)] = m
            if m and m["ret"] is not None:
                log(f"           -> {m['ret']:+.1f}% | {m['ops']} ops | "
                    f"win {m['winpct']:.1f}% | TP {m['tp']} / SL {m['sl']} | liq {m['liq']}")
            else:
                log(f"           -> FALHOU (rc={rc}) ver {out}")

    log("")
    log("=" * 78)
    log("RESUMO")
    log("=" * 78)
    log(f"{'Cenario':<42} {'teste jan-jul':>16} {'holdout jun-jul':>17}")
    log("-" * 76)
    log(f"{'6 ativos · SL fixo 0,5% (BASELINE)':<42} "
        f"{'+2.0%   68ops':>16} {'+0.2%   17ops':>17}")
    for nome, _f, desc in CENARIOS:
        linha = f"{desc:<42}"
        for pkey, _pf, _pd in PERIODOS:
            m = res.get((nome, pkey))
            linha += (f"{m['ret']:>+8.1f}% {m['ops']:>4}ops" if m and m["ret"] is not None
                      else f"{'falhou':>16}")
        log(linha)
    log("=" * 78)
    log("Criterio: so promove ATR se superar o baseline nos DOIS periodos.")


if __name__ == "__main__":
    main()
