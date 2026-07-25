# -*- coding: utf-8 -*-
"""
v6_sweep_k.py — Trader.AI V6: calibra o k do stop adaptativo NA VALIDACAO
==========================================================================

DISCIPLINA (protocolo do projeto): o parametro e escolhido olhando SOMENTE o
split de validacao (H2-2025). O teste (2026) e o holdout virgem (jun-jul/26)
sao usados UMA unica vez, no fim, para confirmar a escolha.

Contexto: no teste, k=6 -> -2,7% | k=7 -> -0,9% | k=8 -> +0,9% (tendencia
crescente). Esses numeros NAO podem escolher o k — servem so de motivacao
para varrer a faixa. A escolha sai da validacao.

Varre k em {6, 8, 10, 12} x {6 ativos, 11 ativos} na VALIDACAO.
Depois confirma o vencedor no teste e no holdout.

Uso: python v6_sweep_k.py
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
LOG  = os.path.join(REL, "v6_sweep_k.log")

MODELO  = "v5_model_b.pth"
HOLDOUT = ("2026-06-01", "2026-07-24")
KS      = [6, 8, 10, 12]
UNIVERSOS = [("6ativos", []), ("11ativos", ["--assets", "ALL"])]


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
            "tp":  g(r"TP\s+:\s*(\d+)\s*ops", int, 0),
            "sl":  g(r"SL\s+:\s*(\d+)\s*ops", int, 0)}


def roda(tag, flags, env):
    out = f"v6_sweep_{tag}.txt"
    with open(os.path.join(REL, out), "w", encoding="utf-8") as fh:
        subprocess.run([PY, "v5_backtest.py", "--model", MODELO] + flags,
                       cwd=BASE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return metricas(os.path.join(REL, out))


def main():
    open(LOG, "w", encoding="utf-8").close()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""

    log("=" * 78)
    log("V6 — SWEEP DO k NA VALIDACAO (H2-2025) — o teste NAO escolhe o k")
    log("=" * 78)

    val = {}
    # Referencia: SL fixo na validacao
    for uni, uflags in UNIVERSOS:
        m = roda(f"val_{uni}_fixo", uflags + ["--val"], env)
        val[(uni, "fixo")] = m
        if m and m["ret"] is not None:
            log(f"  [VAL] {uni:<9} SL fixo 0,5%  -> {m['ret']:+6.1f}% | "
                f"{m['ops']:>3} ops | win {m['win']:.1f}% | TP/SL {m['tp']}/{m['sl']}")

    for uni, uflags in UNIVERSOS:
        for k in KS:
            m = roda(f"val_{uni}_k{k}",
                     uflags + ["--val", "--sl-mode", "atr", "--atr-k", str(k)], env)
            val[(uni, k)] = m
            if m and m["ret"] is not None:
                log(f"  [VAL] {uni:<9} SL {k:>2}xATR      -> {m['ret']:+6.1f}% | "
                    f"{m['ops']:>3} ops | win {m['win']:.1f}% | TP/SL {m['tp']}/{m['sl']}")
            else:
                log(f"  [VAL] {uni:<9} SL {k:>2}xATR      -> FALHOU")

    # Escolha do vencedor pela VALIDACAO
    validos = {kk: v for kk, v in val.items() if v and v["ret"] is not None}
    if not validos:
        log("Nenhum cenario valido.")
        return
    melhor = max(validos, key=lambda kk: validos[kk]["ret"])
    log("")
    log(f"  >> MELHOR NA VALIDACAO: {melhor[0]} / SL {melhor[1]} "
        f"({validos[melhor]['ret']:+.1f}%)")

    # Confirmacao UNICA no teste e no holdout
    log("")
    log("  CONFIRMACAO (uso unico do teste e do holdout):")
    uni = melhor[0]
    uflags = dict(UNIVERSOS)[uni]
    kflags = ([] if melhor[1] == "fixo"
              else ["--sl-mode", "atr", "--atr-k", str(melhor[1])])
    for pnome, pflags in [("teste jan-jul/26", []),
                          ("holdout jun-jul/26",
                           ["--from", HOLDOUT[0], "--to", HOLDOUT[1]])]:
        m = roda(f"conf_{uni}_{melhor[1]}_{pnome.split()[0]}",
                 uflags + kflags + pflags, env)
        if m and m["ret"] is not None:
            log(f"    {pnome:<20} -> {m['ret']:+6.1f}% | {m['ops']:>3} ops | "
                f"win {m['win']:.1f}% | TP/SL {m['tp']}/{m['sl']}")
        else:
            log(f"    {pnome:<20} -> FALHOU")

    log("")
    log("  Baseline atual em producao: 6 ativos + SL fixo = "
        "+2,0% (teste) / +0,2% (holdout)")
    log("  So promove se superar o baseline NOS DOIS periodos.")
    log("=" * 78)


if __name__ == "__main__":
    main()
