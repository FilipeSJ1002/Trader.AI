# -*- coding: utf-8 -*-
"""
v6_ablacao.py — Trader.AI V6: quanto do resultado vem da REDE NEURAL?
======================================================================

O experimento mais importante do projeto.

MOTIVACAO (medido em 25/07/2026 com v6_edge_por_faixa.py):
  Edge direcional do modelo V5.9-B (FAVOR/(FAVOR+CONTRA), 0,50 = aleatorio):
                        TESTE 2026    VALIDACAO H2-2025
      geral (>=0,52)       0,529          0,493
      faixa 2x             0,538          0,498
      faixa 5x             0,515          0,456   <- anti-preditivo!
      faixa 20x            0,594          0,402   <- anti-preditivo!
  Ou seja: a NN nao tem edge consistente entre periodos, e nas faixas de
  ALTA confianca ela chega a ser pior que uma moeda.

PERGUNTA: entao de onde vem o +2,0% do backtest no teste 2026?
HIPOTESE: do FILTRO DE REGIME (SMA 24h). Em 2026 o BTC caiu 26,8% — um filtro
          que so permite SHORT em tendencia de baixa lucra sozinho.

TESTE: rodar a estrategia identica, mas com a NN NEUTRALIZADA (--ablacao sem_nn:
       todo candidato do V1 passa com confianca fixa). Se o resultado for igual
       ou melhor, a rede neural NAO esta agregando valor.

Uso: python v6_ablacao.py
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
LOG  = os.path.join(REL, "v6_ablacao.log")

MODELO  = "v5_model_b.pth"
HOLDOUT = ("2026-06-01", "2026-07-24")

CENARIOS = [
    ("com_nn",  [],                      "COM rede neural (sistema atual)"),
    ("sem_nn",  ["--ablacao", "sem_nn"], "SEM rede neural (so V1 + regime)"),
]
PERIODOS = [
    ("val",     ["--val"],                                    "validacao H2-2025"),
    ("teste",   [],                                           "teste jan-jul/26"),
    ("holdout", ["--from", HOLDOUT[0], "--to", HOLDOUT[1]],   "holdout jun-jul/26"),
]


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
    return {"ret": g(r"Capital final.*?\(([+-][\d.]+)%\)"),
            "ops": g(r"Total de operacoes\s*:\s*(\d+)", int),
            "win": g(r"Vitorias\s*:\s*\d+\s*\(([\d.]+)%\)"),
            "liq": g(r"Liquidacoes\s*:\s*(\d+)", int)}


def main():
    open(LOG, "w", encoding="utf-8").close()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""

    log("=" * 76)
    log("V6 — ABLACAO DA REDE NEURAL: ela agrega valor?")
    log("=" * 76)

    res = {}
    for cnome, cflags, cdesc in CENARIOS:
        for pnome, pflags, pdesc in PERIODOS:
            out = f"v6_abl_{cnome}_{pnome}.txt"
            log(f"[RODANDO] {cdesc} · {pdesc}")
            with open(os.path.join(REL, out), "w", encoding="utf-8") as fh:
                subprocess.run([PY, "v5_backtest.py", "--model", MODELO]
                               + cflags + pflags,
                               cwd=BASE, env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
            m = metricas(os.path.join(REL, out))
            res[(cnome, pnome)] = m
            if m and m["ret"] is not None:
                log(f"           -> {m['ret']:+.1f}% | {m['ops']} ops | "
                    f"win {m['win']:.1f}% | liq {m['liq']}")
            else:
                log(f"           -> FALHOU (ver {out})")

    log("")
    log("=" * 76)
    log("RESULTADO — a rede neural agrega valor?")
    log("=" * 76)
    log(f"  {'Periodo':<22} {'COM NN':>18} {'SEM NN':>18} {'delta':>10}")
    log("  " + "-" * 70)
    for pnome, _pf, pdesc in PERIODOS:
        c = res.get(("com_nn", pnome)); s = res.get(("sem_nn", pnome))
        if not c or not s or c["ret"] is None or s["ret"] is None:
            log(f"  {pdesc:<22} {'(falhou)':>18}")
            continue
        d = c["ret"] - s["ret"]
        log(f"  {pdesc:<22} {c['ret']:>+9.1f}% {c['ops']:>4}ops "
            f"{s['ret']:>+9.1f}% {s['ops']:>4}ops {d:>+9.1f}pp")
    log("=" * 76)
    log("  delta > 0  -> a NN agrega (justifica o custo de treinar/manter)")
    log("  delta ~ 0  -> a NN e decorativa; o valor esta no V1 + regime")
    log("  delta < 0  -> a NN ATRAPALHA (filtra sinais bons)")


if __name__ == "__main__":
    main()
