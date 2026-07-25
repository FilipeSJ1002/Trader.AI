# -*- coding: utf-8 -*-
"""
v6_veredito.py — Trader.AI V6: a V6 supera a V5.9? (comparacao lado a lado)
============================================================================

Roda quando o treino V6 terminar. Compara V6 (26 features) vs V5.9-B (18)
usando EXATAMENTE os mesmos periodos, estrategia e parametros de risco.
So o modelo/features mudam.

Etapas:
  1. CALIBRACAO   — a metrica-alvo da V6: quantas amostras com dir_conf >= 0.62
                    e com que precisao real (FAVOR vs CONTRA) em cada faixa.
  2. BACKTESTS    — validacao (H2-2025) + teste (2026) + holdout virgem (jun-jul/26)
  3. VEREDITO     — tabela comparativa e criterio objetivo de promocao.

Criterio de promocao da V6 (definido ANTES de ver os numeros):
  (a) sinais dir_conf>=0.62 na validacao: V6 > V5.9  E
  (b) razao FAVOR/CONTRA na faixa >=0.62: V6 >= V5.9  E
  (c) retorno no teste 2026: V6 >= V5.9 - 0.5pp (nao pode piorar materialmente)
Se (a) e (b) sim mas (c) nao, o ganho e de sinal mas nao de execucao -> investigar
parametros de risco antes de descartar.

Uso:  python v6_veredito.py            (usa CPU; --gpu para acelerar se livre)
"""
import os
import re
import sys
import argparse
import subprocess
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
REL  = os.path.join(BASE, "relatorios")   # saidas ficam concentradas aqui (gitignored)
os.makedirs(REL, exist_ok=True)
PY   = sys.executable
LOG  = os.path.join(REL, "v6_veredito.log")

HOLDOUT = ("2026-06-01", "2026-07-24")

MODELOS = {
    "V5.9-B": dict(model="v5_model_b.pth", featset="v5",
                   data="data_v5a", y_dir="data_v5b"),
    "V6.0":   dict(model="v6_model.pth",   featset="v6",
                   data="data_v6",  y_dir=None),
}

PERIODOS = [
    ("val",       ["--val"],                                       "Validacao H2-2025"),
    # Periodo historico exato (comparavel ao baseline +1,8% documentado)
    ("teste_hist", ["--from", "2026-01-01", "--to", "2026-05-31"], "Teste jan-mai/26 (historico)"),
    # Tudo que existe de 2026 (jan-jul) — com os dados frescos de 25/07
    ("teste_full", [],                                             "Teste 2026 completo (jan-jul)"),
    ("holdout",   ["--from", HOLDOUT[0], "--to", HOLDOUT[1]],      "Holdout virgem jun-jul/26"),
]


def log(msg=""):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def run(cmd, outfile=None, env=None):
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    if outfile:
        with open(os.path.join(REL, outfile), "w", encoding="utf-8") as fh:
            return subprocess.run(cmd, cwd=BASE, env=e, stdout=fh,
                                  stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=BASE, env=e).returncode


def metricas_bt(path):
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
        "btcpct": g(r"Hold de BTC.*?\(([+-][\d.]+)%\)"),
        "lev5":   g(r"5x FUTUROS\s*:\s*(\d+)\s*ops", int, 0),
    }


def metricas_cal(path):
    """Le a saida do v6_calibracao.py."""
    if not os.path.exists(path):
        return None
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"dir_conf >= 0\.62:\s*(\d+)\s*\(([\d.]+)%", t)
    faixa = re.search(r"0\.62-0\.67.*?\s(\d+)\s+[\d.]+%\s+([\d.]+)%\s+"
                      r"([\d.]+)%\s+([\d.]+)%\s+([\d.]+)", t)
    return {
        "n62":    int(m.group(1)) if m else None,
        "pct62":  float(m.group(2)) if m else None,
        "fc_5x":  float(faixa.group(5)) if faixa else None,
        "favor":  float(faixa.group(2)) if faixa else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Usa GPU (padrao CPU)")
    ap.add_argument("--split-cal", default="val", choices=["val", "test"])
    a = ap.parse_args()

    open(LOG, "w", encoding="utf-8").close()
    env = {} if a.gpu else {"CUDA_VISIBLE_DEVICES": ""}

    if not os.path.exists(os.path.join(BASE, "v6_model.pth")):
        log("v6_model.pth ainda nao existe — o treino precisa terminar antes.")
        return

    log("=" * 76)
    log("  V6 VEREDITO — V6.0 (26 features) vs V5.9-B (18 features)")
    log(f"  {datetime.now():%Y-%m-%d %H:%M} | device: {'GPU' if a.gpu else 'CPU'}")
    log("=" * 76)

    # ---------- 1. Calibracao ----------
    log("\n[1/2] CALIBRACAO — volume e qualidade dos sinais de alta conviccao\n")
    cal = {}
    for nome, cfg in MODELOS.items():
        out = f"v6_cal_{nome.replace('.', '_')}.txt"
        cmd = [PY, "v6_calibracao.py", "--model", cfg["model"],
               "--data", cfg["data"], "--split", a.split_cal]
        if cfg["y_dir"]:
            cmd += ["--y-dir", cfg["y_dir"]]
        if a.gpu:
            cmd += ["--gpu"]
        log(f"  rodando {nome}...")
        run(cmd, outfile=out, env=env)
        cal[nome] = metricas_cal(os.path.join(REL, out))

    log(f"\n  {'Modelo':<10} {'amostras>=0.62':>15} {'% do split':>11} "
        f"{'FAVOR na 5x':>12} {'F/C na 5x':>10}")
    log("  " + "-" * 62)
    for nome, m in cal.items():
        if not m or m["n62"] is None:
            log(f"  {nome:<10} {'(falhou)':>15}")
            continue
        log(f"  {nome:<10} {m['n62']:>15,} {m['pct62']:>10.2f}% "
            f"{(m['favor'] or 0):>11.1f}% {(m['fc_5x'] or 0):>10.2f}")

    # ---------- 2. Backtests ----------
    log("\n[2/2] BACKTESTS — mesma estrategia, mesmos periodos\n")
    bt = {}
    for nome, cfg in MODELOS.items():
        for chave, flags, desc in PERIODOS:
            out = f"v6_bt_{nome.replace('.', '_')}_{chave}.txt"
            cmd = ([PY, "v5_backtest.py", "--model", cfg["model"],
                    "--featset", cfg["featset"]] + flags)
            log(f"  {nome} · {desc}...")
            run(cmd, outfile=out, env=env)
            bt[(nome, chave)] = metricas_bt(os.path.join(REL, out))

    log("\n" + "=" * 76)
    log("  RESULTADO COMPARATIVO")
    log("=" * 76)
    log(f"  {'Periodo':<26} {'V5.9-B':>22} {'V6.0':>22}")
    log("  " + "-" * 72)
    for chave, _f, desc in PERIODOS:
        linha = f"  {desc:<26}"
        for nome in MODELOS:
            m = bt.get((nome, chave))
            if not m or m["ret"] is None:
                linha += f"{'(falhou)':>22}"
            else:
                linha += (f"{m['ret']:>+9.1f}% {m['ops']:>4}ops "
                          f"w{m['winpct']:>4.0f}%")
        log(linha)
    log("=" * 76)

    # ---------- Veredito ----------
    log("\n  CRITERIO DE PROMOCAO (definido antes dos numeros):")
    c5, c6 = cal.get("V5.9-B"), cal.get("V6.0")
    b5, b6 = bt.get(("V5.9-B", "teste_full")), bt.get(("V6.0", "teste_full"))
    ok = []
    if c5 and c6 and None not in (c5["n62"], c6["n62"]):
        a_ok = c6["n62"] > c5["n62"]
        ok.append(a_ok)
        log(f"   (a) mais sinais >=0.62 .......... {'SIM' if a_ok else 'NAO'} "
            f"({c5['n62']:,} -> {c6['n62']:,})")
    if c5 and c6 and None not in (c5["fc_5x"], c6["fc_5x"]):
        b_ok = c6["fc_5x"] >= c5["fc_5x"]
        ok.append(b_ok)
        log(f"   (b) qualidade F/C na faixa 5x ... {'SIM' if b_ok else 'NAO'} "
            f"({c5['fc_5x']:.2f} -> {c6['fc_5x']:.2f})")
    if b5 and b6 and None not in (b5["ret"], b6["ret"]):
        c_ok = b6["ret"] >= b5["ret"] - 0.5
        ok.append(c_ok)
        log(f"   (c) retorno no teste nao piora ... {'SIM' if c_ok else 'NAO'} "
            f"({b5['ret']:+.1f}% -> {b6['ret']:+.1f}%)")
    if ok:
        log(f"\n  >> VEREDITO: {'PROMOVER V6' if all(ok) else 'NAO PROMOVER (investigar)'}")
    log("=" * 76)


if __name__ == "__main__":
    main()
