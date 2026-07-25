# -*- coding: utf-8 -*-
"""
v6_exp_curvas.py — Trader.AI V6: a curva de alavancagem esta bem calibrada?
============================================================================

CONTEXTO (25/07/2026):
  A analise v6_edge_por_faixa.py sugeriu que a curva historica esta
  anti-correlacionada com o edge (alavanca 5x onde o edge e menor).
  MAS aquela medicao foi feita em janelas ALEATORIAS do mercado — e a
  ablacao provou que esse nao e o contexto de uso da rede (ver licao 20b:
  medir no contexto de uso, nao em abstrato).

  Este experimento faz a medicao CERTA: roda o backtest completo — onde a
  rede so opera sobre candidatos do V1 — variando apenas a curva.

CURVAS (definidas em v5_backtest.CURVAS_LEV):
  v59   historica: 0,52->1x  0,57->2x  0,62->5x
  edge  realinhada: 0,52->2x  0,57->5x  0,62->1x
  pico  so a faixa 0,57-0,62 (5x), ignora o resto
  flat2 tudo 2x   |  flat1 tudo 1x (sem alavancagem discriminada)

DISCIPLINA: escolhe pela VALIDACAO; confirma o vencedor UMA vez no teste
e no holdout virgem.

Uso: python v6_exp_curvas.py
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
LOG  = os.path.join(REL, "v6_exp_curvas.log")

MODELO  = "v5_model_b.pth"
HOLDOUT = ("2026-06-01", "2026-07-24")
CURVAS  = ["v59", "edge", "pico", "flat2", "flat1"]


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


def roda(tag, flags, env):
    out = f"v6_curva_{tag}.txt"
    with open(os.path.join(REL, out), "w", encoding="utf-8") as fh:
        subprocess.run([PY, "v5_backtest.py", "--model", MODELO] + flags,
                       cwd=BASE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return metricas(os.path.join(REL, out))


def main():
    open(LOG, "w", encoding="utf-8").close()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""

    log("=" * 72)
    log("V6 — CURVAS DE ALAVANCAGEM (medidas no contexto de uso: backtest)")
    log("=" * 72)

    val = {}
    for c in CURVAS:
        m = roda(f"val_{c}", ["--val", "--lev-curve", c], env)
        val[c] = m
        if m and m["ret"] is not None:
            log(f"  [VAL] curva {c:<6} -> {m['ret']:+6.1f}% | {m['ops']:>3} ops | "
                f"win {m['win']:.1f}% | liq {m['liq']}")
        else:
            log(f"  [VAL] curva {c:<6} -> FALHOU")

    validas = {c: m for c, m in val.items() if m and m["ret"] is not None}
    if not validas:
        log("Nenhuma curva valida.")
        return
    melhor = max(validas, key=lambda c: validas[c]["ret"])
    log("")
    log(f"  >> MELHOR NA VALIDACAO: curva '{melhor}' ({validas[melhor]['ret']:+.1f}%)")
    log(f"     (historica 'v59' = {validas.get('v59', {}).get('ret', float('nan')):+.1f}%)")

    log("")
    log("  CONFIRMACAO do vencedor (uso unico do teste e do holdout):")
    for pnome, pflags in [("teste jan-jul/26", []),
                          ("holdout jun-jul/26",
                           ["--from", HOLDOUT[0], "--to", HOLDOUT[1]])]:
        m = roda(f"conf_{melhor}_{pnome.split()[0]}",
                 ["--lev-curve", melhor] + pflags, env)
        if m and m["ret"] is not None:
            log(f"    {pnome:<20} -> {m['ret']:+6.1f}% | {m['ops']:>3} ops | "
                f"win {m['win']:.1f}% | liq {m['liq']}")
        else:
            log(f"    {pnome:<20} -> FALHOU")

    log("")
    log("  Baseline em producao (curva v59): +2,0% teste | +0,2% holdout | -2,0% val")
    log("  So promove se superar o baseline NOS DOIS periodos de confirmacao.")
    log("=" * 72)


if __name__ == "__main__":
    main()
