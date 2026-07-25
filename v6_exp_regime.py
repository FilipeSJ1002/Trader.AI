# -*- coding: utf-8 -*-
"""
v6_exp_regime.py — Trader.AI V6: alavancagem em funcao do REGIME
=================================================================

HIPOTESE (evidencia de 25/07/2026, Frente 6):
  A alavancagem so compensa quando ha TENDENCIA FORTE.
      bear forte (teste 2026, BTC -26,8%): curva alavancada +2,0% vs +0,3%
      lateral (validacao H2-2025):         curva alavancada -2,0% vs +0,5%
      sem alavancagem alguma (flat1) supera a de producao em 1,2 pp na validacao
  Motivo economico: taxas escalam com o notional (0,04% x margem x alavancagem).
  Com edge ~0,50, alavancar em mercado lateral so paga mais taxa pela mesma
  expectativa — e multiplica a variancia.

SOLUCAO PROPOSTA: alavancagem = f(forca da tendencia) x f(confianca)
  forca = |preco - SMA24h| / SMA24h, normalizada pela mediana do proprio ativo
          (1,0 = distancia tipica; >1,5 = tendencia esticada/forte)
  curva "regime"      -> forte: escada 1x/2x/5x | lateral: 1x
  curva "regime_pico" -> forte: 5x na faixa de melhor edge | lateral: 1x

Varre forca_min em {1.2, 1.5, 2.0} NA VALIDACAO; confirma o vencedor uma
unica vez no teste e no holdout virgem.

Uso: python v6_exp_regime.py
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
LOG  = os.path.join(REL, "v6_exp_regime.log")

MODELO  = "v5_model_b.pth"
HOLDOUT = ("2026-06-01", "2026-07-24")

# (curva, forca_min)
COMBOS = [("regime", 1.2), ("regime", 1.5), ("regime", 2.0),
          ("regime_pico", 1.5), ("regime_pico", 2.0)]

# Referencias ja medidas (curva v59 = producao)
REF = {"val": -2.0, "teste": +2.0, "holdout": +0.2}


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
    out = f"v6_regime_{tag}.txt"
    with open(os.path.join(REL, out), "w", encoding="utf-8") as fh:
        subprocess.run([PY, "v5_backtest.py", "--model", MODELO] + flags,
                       cwd=BASE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return metricas(os.path.join(REL, out))


def main():
    open(LOG, "w", encoding="utf-8").close()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["CUDA_VISIBLE_DEVICES"] = ""

    log("=" * 74)
    log("V6 — ALAVANCAGEM POR REGIME (alavanca so em tendencia forte)")
    log(f"Producao (curva v59): val {REF['val']:+.1f}% | teste {REF['teste']:+.1f}% "
        f"| holdout {REF['holdout']:+.1f}%")
    log("=" * 74)

    val = {}
    for curva, fmin in COMBOS:
        tag = f"val_{curva}_{fmin}"
        m = roda(tag, ["--val", "--lev-curve", curva, "--forca-min", str(fmin)], env)
        val[(curva, fmin)] = m
        if m and m["ret"] is not None:
            log(f"  [VAL] {curva:<12} forca>={fmin:<4} -> {m['ret']:+6.1f}% | "
                f"{m['ops']:>3} ops | win {m['win']:.1f}%")
        else:
            log(f"  [VAL] {curva:<12} forca>={fmin:<4} -> FALHOU")

    validos = {k: v for k, v in val.items() if v and v["ret"] is not None}
    if not validos:
        log("Nenhum combo valido.")
        return
    melhor = max(validos, key=lambda k: validos[k]["ret"])
    log("")
    log(f"  >> MELHOR NA VALIDACAO: {melhor[0]} forca>={melhor[1]} "
        f"({validos[melhor]['ret']:+.1f}%) | producao: {REF['val']:+.1f}%")

    log("")
    log("  CONFIRMACAO (uso unico do teste e do holdout):")
    resultados = {}
    for pnome, pkey, pflags in [("teste jan-jul/26", "teste", []),
                                ("holdout jun-jul/26", "holdout",
                                 ["--from", HOLDOUT[0], "--to", HOLDOUT[1]])]:
        m = roda(f"conf_{melhor[0]}_{melhor[1]}_{pkey}",
                 ["--lev-curve", melhor[0], "--forca-min", str(melhor[1])] + pflags,
                 env)
        resultados[pkey] = m
        if m and m["ret"] is not None:
            log(f"    {pnome:<20} -> {m['ret']:+6.1f}% | {m['ops']:>3} ops | "
                f"win {m['win']:.1f}% | producao {REF[pkey]:+.1f}%")
        else:
            log(f"    {pnome:<20} -> FALHOU")

    log("")
    log("  VEREDITO:")
    ok = []
    for pkey in ["teste", "holdout"]:
        m = resultados.get(pkey)
        if m and m["ret"] is not None:
            passou = m["ret"] >= REF[pkey]
            ok.append(passou)
            log(f"    {pkey:<8}: {m['ret']:+.1f}% vs {REF[pkey]:+.1f}% "
                f"-> {'PASSA' if passou else 'nao passa'}")
    if ok:
        log(f"\n  >> {'PROMOVER alavancagem por regime' if all(ok) else 'MANTER curva v59'}")
    log("=" * 74)


if __name__ == "__main__":
    main()
