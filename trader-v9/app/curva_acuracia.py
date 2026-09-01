# -*- coding: utf-8 -*-
"""
app/curva_acuracia.py — a partir de qual acuracia isto vale a pena?
====================================================================

A entrega real da Sprint 1. Roda a arquitetura completa com oraculos de
acuracia crescente e devolve a curva: quanto rende cada nivel de acerto.

O ponto onde a curva cruza o zero e o ALVO do classificador da Sprint 2. Se
esse ponto estiver acima de ~55%, o projeto nao tem alvo alcancavel — prever
direcao diaria de cripto com acuracia sustentada acima disso nao e coisa que
exista.

Cada nivel roda com varias sementes (o sorteio do erro muda) e varias fases
(a grade de avaliacao muda), e o resultado sai com margem de erro.
"""
import sys
import time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import numpy as np

from avaliacao.metricas import comprar_e_segurar, medir
from avaliacao.replay import replay
from dados.fonte import FonteParquet, ler_config
from execucao.papel import CorretoraPapel
from execucao.risco import Risco
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.tipos import Regime
from oraculo.ruidoso import OraculoRuidoso

NIVEIS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
SEMENTES = [1, 2, 3]
FASES = [0, 7]
DE, ATE = datetime(2023, 1, 1), datetime(2026, 7, 25)
CAPITAL = 5000.0


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)
    ref = comprar_e_segurar(h, DE, ATE, CAPITAL)

    total = len(NIVEIS) * len(SEMENTES) * len(FASES)
    print(f"{DE:%Y-%m-%d} a {ATE:%Y-%m-%d} | {total} rodadas "
          f"({len(NIVEIS)} niveis x {len(SEMENTES)} sementes x "
          f"{len(FASES)} fases)")
    print(f"Comprar e segurar no periodo: {ref.retorno*100:+.1f}%\n", flush=True)

    print(f"  {'acuracia':<10} {'media':>10} {'erro':>8} {'t':>7} "
          f"{'pior':>10} {'melhor':>10} {'queda':>8} {'ops':>7}")
    print("  " + "-" * 76, flush=True)

    linhas = []
    feito = 0
    for p in NIVEIS:
        retornos, quedas, ops = [], [], []
        for semente in SEMENTES:
            for fase in FASES:
                c = CorretoraPapel(CAPITAL)
                r = replay(
                    h,
                    {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                    OraculoRuidoso(h, acuracia=p, semente=semente),
                    c, Risco(), DE, ATE, a_cada=15,
                    referencia=ativos[0], fase=fase,
                )
                m = medir(r)
                retornos.append(m.retorno)
                quedas.append(m.rebaixamento)
                ops.append(m.operacoes)
                feito += 1

        a = np.array(retornos)
        erro = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        t = a.mean() / erro if erro > 0 else 0.0
        linhas.append((p, a.mean(), erro, t, a.min(), a.max(),
                       float(np.mean(quedas)), float(np.mean(ops))))
        print(f"  {p:>8.0%}   {a.mean()*100:>+9.1f}% {erro*100:>7.1f} "
              f"{t:>+7.2f} {a.min()*100:>+9.1f}% {a.max()*100:>+9.1f}% "
              f"{np.mean(quedas)*100:>7.1f}% {np.mean(ops):>7.0f}"
              f"   [{feito}/{total}]", flush=True)

    _veredito(linhas, ref)


def _veredito(linhas, ref) -> None:
    print("\n" + "=" * 80)
    print("  O ALVO DO CLASSIFICADOR DA SPRINT 2")
    print("=" * 80)

    cruz_zero = _cruzamento(linhas, 0.0)
    cruz_bh = _cruzamento(linhas, ref.retorno)

    print(f"\n  Acuracia necessaria para nao perder dinheiro : "
          f"{_fmt(cruz_zero)}")
    print(f"  Acuracia necessaria para bater comprar-e-segurar "
          f"({ref.retorno*100:+.0f}%): {_fmt(cruz_bh)}")

    print("\n  Referencia honesta: prever a direcao diaria de cripto com")
    print("  acuracia sustentada acima de ~55% nao e resultado que exista na")
    print("  literatura seria. 52-53% ja seria excepcional.\n")

    if cruz_zero is None:
        print("  NENHUM nivel de acuracia — nem 100% — para de perder dinheiro.")
        print("  O problema nao e a previsao: sao os motores ou o custo.")
    elif cruz_zero > 0.60:
        print(f"  ALVO INALCANCAVEL. Seria preciso {cruz_zero:.0%} de acerto")
        print("  diario so para empatar. Nao construir o classificador.")
    else:
        print(f"  ALVO PLAUSIVEL: {cruz_zero:.0%}. Vale construir o classificador")
        print("  e medir sua acuracia real ANTES de liga-lo a operacao.")
    print("=" * 80)


def _cruzamento(linhas, alvo):
    """A menor acuracia cuja media supera `alvo` (interpolando entre niveis)."""
    for (p0, m0, *_), (p1, m1, *_) in zip(linhas, linhas[1:]):
        if m0 <= alvo < m1:
            return p0 + (p1 - p0) * (alvo - m0) / (m1 - m0)
    if linhas and linhas[0][1] > alvo:
        return linhas[0][0]
    return None


def _fmt(x):
    return "nunca" if x is None else f"{x:.0%}"


if __name__ == "__main__":
    main()
