# -*- coding: utf-8 -*-
"""
app/figuras.py — as figuras do artigo
======================================

Gera dois gráficos em PNG, em estilo sóbrio e legível em impressão
monocromática (linhas distinguíveis por traço, não só por cor).

  Figura A — curva de acurácia: quanto rende cada nível de acerto do oráculo,
             com os dois limiares marcados e a posição do classificador real
  Figura B — curvas de capital: as configurações contra comprar-e-segurar

Uso:
  python app/figuras.py --saida <diretorio>
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from avaliacao.metricas import comprar_e_segurar
from avaliacao.replay import replay
from dados.fonte import FonteParquet, ler_config
from execucao.papel import CorretoraPapel
from execucao.risco import Risco
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.tipos import Regime
from oraculo.ruidoso import OraculoLento, OraculoRuidoso

# Medições de 26 e 31/08/2026 (app/curva_acuracia.py)
CURVA_1D = {50: -3.2, 55: 153.4, 60: 535.7, 65: 1546.0, 70: 3909.1}
CURVA_3D = {50: -24.7, 52: -1.7, 54: 16.6, 56: 52.8,
            58: 86.0, 60: 129.9, 65: 290.2, 70: 532.2}
ERRO_3D = {50: 7.9, 52: 10.3, 54: 8.2, 56: 12.6,
           58: 17.1, 60: 30.3, 65: 38.5, 70: 64.5}

CLASSIFICADOR = 53.69          # acurácia balanceada medida, horizonte 3 dias
BUY_HOLD = 215.6               # retorno de comprar-e-segurar no mesmo período
DE, ATE = datetime(2023, 1, 1), datetime(2026, 7, 25)
CAPITAL = 5000.0


def _estilo() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
    })


def figura_acuracia(caminho: str) -> None:
    """Quanto rende cada nível de acurácia, com os limiares marcados."""
    _estilo()
    fig, ax = plt.subplots(figsize=(6.3, 3.9))

    x3 = np.array(sorted(CURVA_3D))
    y3 = np.array([CURVA_3D[k] for k in x3])
    e3 = np.array([ERRO_3D[k] for k in x3])

    ax.errorbar(x3, y3, yerr=e3, fmt="o-", color="#1a1a1a", linewidth=1.6,
                markersize=4, capsize=3, elinewidth=0.9,
                label="Regime de 3 dias (medido)")

    x1 = np.array(sorted(CURVA_1D))
    y1 = np.array([CURVA_1D[k] for k in x1])
    ax.plot(x1, y1, "s--", color="#888888", linewidth=1.2, markersize=3.5,
            label="Regime de 1 dia (referência)")

    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.axhline(BUY_HOLD, color="#555555", linewidth=0.9, linestyle=":")
    ax.text(50.2, BUY_HOLD + 18, f"comprar e segurar (+{BUY_HOLD:.0f}%)",
            fontsize=7.5, color="#444444")

    # O que temos, e o que seria preciso.
    ax.axvline(CLASSIFICADOR, color="#B03A2E", linewidth=1.3)
    ax.text(CLASSIFICADOR + 0.25, 430, f"classificador\nmedido\n{CLASSIFICADOR:.2f}%",
            fontsize=7.5, color="#B03A2E", va="top")
    ax.axvline(63, color="#1D6F4E", linewidth=1.3, linestyle="--")
    ax.text(63 + 0.25, 430, "necessário\npara igualar\n63%", fontsize=7.5,
            color="#1D6F4E", va="top")

    ax.set_xlabel("Acurácia do oráculo de regime (%)")
    ax.set_ylabel("Retorno acumulado em 3,5 anos (%)")
    ax.set_xlim(49.5, 70.5)
    ax.set_ylim(-80, 560)
    # Convencao brasileira: virgula decimal, e marcas nos niveis medidos.
    ax.set_xticks([50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70])
    ax.set_yticks([-100, 0, 100, 200, 300, 400, 500])
    ax.set_yticklabels([f"{v:+d}".replace("+0", "0") for v in
                        [-100, 0, 100, 200, 300, 400, 500]])
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {caminho}")


def figura_capital(caminho: str, historicos, ativos) -> None:
    """Curvas de capital das configurações contra comprar-e-segurar."""
    _estilo()
    motores = {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()}

    cenarios = [
        ("Oráculo de 3 dias a 53,7% (o que temos)",
         lambda: OraculoLento(historicos, passo=3, acuracia=0.537, semente=1),
         "-", "#1a1a1a", 1.8),
        ("Oráculo de 3 dias a 60% (hipotético)",
         lambda: OraculoLento(historicos, passo=3, acuracia=0.60, semente=1),
         "--", "#666666", 1.3),
        ("Oráculo diário a 50% (sorteio)",
         lambda: OraculoRuidoso(historicos, acuracia=0.50, semente=1),
         ":", "#999999", 1.3),
    ]

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for rotulo, fabrica, traco, cor, largura in cenarios:
        c = CorretoraPapel(CAPITAL)
        r = replay(historicos, motores, fabrica(), c, Risco(),
                   DE, ATE, a_cada=15, referencia=ativos[0])
        ts = [t for t, _ in r.curva]
        v = np.array([x for _, x in r.curva]) / CAPITAL
        ax.plot(ts, v, traco, color=cor, linewidth=largura, label=rotulo)
        print(f"  {rotulo}: {r.retorno*100:+.1f}%", flush=True)

    # Comprar e segurar, com a mesma base.
    curvas = []
    for h in historicos.values():
        i0, i1 = h.indice_de(DE), h.indice_de(ATE)
        s = h.em(i1).serie("fechamento")[i0:i1 + 1]
        curvas.append(s / s[0])
    n = min(len(c) for c in curvas)
    bh = np.mean([c[:n] for c in curvas], axis=0)
    idx = historicos[ativos[0]]
    ts_bh = [t.astype("datetime64[us]").astype(datetime)
             for t in idx.instantes[idx.indice_de(DE):idx.indice_de(DE) + n]]
    ax.plot(ts_bh, bh, "-", color="#1D6F4E", linewidth=1.6,
            label="Comprar e segurar os 6 ativos")

    ax.axhline(1.0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Capital (múltiplo do inicial)")
    ax.set_xlabel("Período")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 3, 5])
    ax.set_yticklabels(["0,5x", "1x", "2x", "3x", "5x"])
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {caminho}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=".")
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)

    figura_acuracia(os.path.join(a.saida, "fig_acuracia.png"))

    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print("Carregando ativos para as curvas de capital...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)
    ref = comprar_e_segurar(h, DE, ATE, CAPITAL)
    print(f"  comprar e segurar: {ref.retorno*100:+.1f}%", flush=True)
    figura_capital(os.path.join(a.saida, "fig_capital.png"), h, ativos)


if __name__ == "__main__":
    main()
