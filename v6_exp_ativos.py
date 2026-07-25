# -*- coding: utf-8 -*-
"""
v6_exp_ativos.py — Trader.AI V6: isola o efeito "mais ativos" e testa o holdout virgem
=======================================================================================

Roda em CPU (nao disputa a GPU com o treino) o modelo V5.9 campeao em 4 cenarios.
Como o modelo so ve features normalizadas/adimensionais, ele PODE operar ativos
que nunca viu no treino — este experimento mede se isso vale a pena.

  1. teste 2026 · 6 ativos   -> baseline conhecido (+1,8%)
  2. teste 2026 · 11 ativos  -> efeito isolado de ampliar o universo
  3. holdout jun-jul/2026 · 6 ativos   -> dados VIRGENS (nenhuma decisao os tocou)
  4. holdout jun-jul/2026 · 11 ativos  -> idem, universo ampliado

O cenario 3/4 e a avaliacao mais honesta possivel: esses candles foram baixados
DEPOIS de todas as decisoes de projeto terem sido tomadas.

Uso:  python v6_exp_ativos.py
Saidas: v6_bt_<cenario>.txt + resumo no console/log
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
LOG  = os.path.join(REL, "v6_exp_ativos.log")

MODELO  = "v5_model_b.pth"
HOLDOUT = ("2026-06-01", "2026-07-24")   # virgem: baixado hoje, apos as decisoes

CENARIOS = [
    ("teste2026_6ativos",   ["--model", MODELO], "Teste 2026 (jan-mai) · 6 ativos"),
    ("teste2026_11ativos",  ["--model", MODELO, "--assets", "ALL"],
     "Teste 2026 (jan-mai) · 11 ativos"),
    ("holdout_6ativos",     ["--model", MODELO,
                             "--from", HOLDOUT[0], "--to", HOLDOUT[1]],
     "HOLDOUT VIRGEM jun-jul/2026 · 6 ativos"),
    ("holdout_11ativos",    ["--model", MODELO, "--assets", "ALL",
                             "--from", HOLDOUT[0], "--to", HOLDOUT[1]],
     "HOLDOUT VIRGEM jun-jul/2026 · 11 ativos"),
]


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def extrai(path):
    """Puxa as metricas-chave do relatorio de backtest."""
    if not os.path.exists(path):
        return None
    txt = open(path, encoding="utf-8", errors="replace").read()
    def g(pat, cast=float, default=None):
        m = re.search(pat, txt)
        return cast(m.group(1).replace(",", "")) if m else default
    return {
        "cap":    g(r"Capital final\s*:\s*\$\s*([\d,.]+)"),
        "ret":    g(r"Capital final.*?\(([+-][\d.]+)%\)"),
        "ops":    g(r"Total de operacoes\s*:\s*(\d+)", int),
        "wins":   g(r"Vitorias\s*:\s*(\d+)", int),
        "winpct": g(r"Vitorias\s*:\s*\d+\s*\(([\d.]+)%\)"),
        "liq":    g(r"Liquidacoes\s*:\s*(\d+)", int),
        "btc":    g(r"Hold de BTC\s*:\s*\$\s*([\d,.]+)"),
        "btcpct": g(r"Hold de BTC.*?\(([+-][\d.]+)%\)"),
        "lev5":   g(r"5x FUTUROS\s*:\s*(\d+)\s*ops", int, 0),
    }


def main():
    open(LOG, "w", encoding="utf-8").close()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""      # CPU: nao disputa VRAM com o treino

    log("=" * 70)
    log("V6 — EXPERIMENTO: universo de ativos + holdout virgem (modelo V5.9-B)")
    log("=" * 70)

    resultados = []
    for nome, flags, desc in CENARIOS:
        out = f"v6_bt_{nome}.txt"
        log(f"[RODANDO] {desc}")
        t0 = datetime.now()
        with open(os.path.join(REL, out), "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, "v5_backtest.py"] + flags,
                                cwd=BASE, env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = (datetime.now() - t0).total_seconds() / 60
        m = extrai(os.path.join(REL, out))
        if rc != 0 or m is None or m["cap"] is None:
            log(f"  -> FALHOU (rc={rc}) — ver {out}")
            resultados.append((desc, None))
            continue
        log(f"  -> {m['ret']:+.1f}% | {m['ops']} ops | win {m['winpct']:.1f}% "
            f"| liq {m['liq']} | BTC {m['btcpct']:+.1f}% | {dt:.1f}min")
        resultados.append((desc, m))

    log("")
    log("=" * 70)
    log("RESUMO COMPARATIVO")
    log("=" * 70)
    log(f"{'Cenario':<40} {'Retorno':>9} {'Ops':>5} {'Win':>7} {'5x':>4} {'vs BTC':>9}")
    log("-" * 78)
    for desc, m in resultados:
        if m is None:
            log(f"{desc:<40} {'FALHOU':>9}")
            continue
        vs = m["ret"] - m["btcpct"] if m["btcpct"] is not None else float("nan")
        log(f"{desc:<40} {m['ret']:>+8.1f}% {m['ops']:>5} {m['winpct']:>6.1f}% "
            f"{m['lev5']:>4} {vs:>+8.1f}pp")
    log("=" * 70)


if __name__ == "__main__":
    main()
