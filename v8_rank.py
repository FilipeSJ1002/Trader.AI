# -*- coding: utf-8 -*-
"""
v8_rank.py — o ranking das configuracoes, medido pelo simulador de producao
===========================================================================

Cada linha do ranking e uma rodada COMPLETA do v8_simulador: replay minuto a
minuto chamando decidir_entrada() e avaliar_saidas() de v6_ciclo.py. A unica
coisa que muda entre as linhas sao os parametros do modulo — os mesmos que o
bot ao vivo le. Nenhuma logica de decisao e reimplementada aqui.

Criterio de ordenacao: RETORNO EQUIVALENTE MENSAL. Mas retorno sozinho engana —
uma configuracao com +2%/mes e rebaixamento de 60% e pior que uma com +1%/mes e
rebaixamento de 8%, porque a primeira quebra a conta antes do fim. Por isso a
tabela mostra os dois, mais o numero de operacoes: poucas operacoes significa
que o resultado pode ser sorte.

Referencia obrigatoria: COMPRAR E SEGURAR. Se a estrategia nao bate uma carteira
parada, ela nao paga o trabalho nem o risco.
"""
import numpy as np
import pandas as pd

import v6_ciclo
from v5_data_prep import ASSETS


# Cada configuracao sobrescreve os globais que o codigo de producao le.
# O que nao aparece no dicionario fica no valor de producao.
CONFIGS = [
    # -- o que esta rodando hoje -------------------------------------------
    ("PRODUCAO (config B)", {}),

    # -- hipotese 1: o filtro da rede neural nao ajuda ---------------------
    # Medido em 25/08/2026 sobre 6.619 eventos: fora da amostra os sinais que
    # ele vetou renderam +0,1496% e os que aprovou +0,1433%. Aqui o teste e
    # com curva de capital, que e o que decide.
    # ATENCAO: baixar DIRCONF_MIN nao basta — leverage_for zera a entrada
    # abaixo de 0.52 em todas as curvas antigas. Por isso as curvas "_livre".
    ("B sem filtro da rede",        {"DIRCONF_MIN": 0.0,
                                     "LEV_CURVE": "regime_livre"}),
    ("B com filtro exigente 0.60",  {"DIRCONF_MIN": 0.60}),

    # -- hipotese 2: o tamanho do alvo e o que importa ---------------------
    # Foi o que fez a config B vencer a A. A pergunta e se continua valendo
    # quando o alvo cresce mais, ou se satura.
    ("A antiga 1,0%/0,5%/6h",  {"TP_PCT": 0.010, "SL_PCT": 0.005,
                                "MAX_HOLD_MIN": 360}),
    ("Alvo 5%/2,5% / 4 dias",  {"TP_PCT": 0.050, "SL_PCT": 0.025,
                                "MAX_HOLD_MIN": 5760}),
    ("Alvo 8%/4% / 7 dias",    {"TP_PCT": 0.080, "SL_PCT": 0.040,
                                "MAX_HOLD_MIN": 10080}),
    ("Alvo 12%/6% / 14 dias",  {"TP_PCT": 0.120, "SL_PCT": 0.060,
                                "MAX_HOLD_MIN": 20160}),

    # -- hipotese 3: a razao alvo/stop -------------------------------------
    ("B razao 3:1 (3%/1%)",    {"TP_PCT": 0.030, "SL_PCT": 0.010}),
    ("B razao 1:1 (3%/3%)",    {"TP_PCT": 0.030, "SL_PCT": 0.030}),

    # -- hipotese 4: a alavancagem ----------------------------------------
    ("B sem alavancagem",      {"MAX_LEV": 1.0, "LEV_CURVE": "flat1"}),
    ("B alavancagem fixa 2x",  {"MAX_LEV": 2.0, "LEV_CURVE": "flat2"}),
    ("B curva v59",            {"LEV_CURVE": "v59"}),

    # -- hipotese 5: as duas melhores ideias juntas ------------------------
    ("Alvo 8% SEM filtro",     {"TP_PCT": 0.080, "SL_PCT": 0.040,
                                "MAX_HOLD_MIN": 10080, "DIRCONF_MIN": 0.0,
                                "LEV_CURVE": "regime_livre"}),
    ("Alvo 8% SEM filtro 1x",  {"TP_PCT": 0.080, "SL_PCT": 0.040,
                                "MAX_HOLD_MIN": 10080, "DIRCONF_MIN": 0.0,
                                "MAX_LEV": 1.0, "LEV_CURVE": "flat1_livre"}),
]

PARAMETROS = ["SL_PCT", "TP_PCT", "MAX_LEV", "MAX_HOLD_MIN",
              "LEV_CURVE", "FORCA_MIN", "DIRCONF_MIN"]


def _aplicar(mudancas, originais):
    for k in PARAMETROS:
        setattr(v6_ciclo, k, mudancas.get(k, originais[k]))


def comprar_e_segurar(dados, de, ate, capital):
    """Carteira de peso igual, comprada o periodo inteiro. A referencia."""
    curvas = []
    for sym, d in dados.items():
        if sym not in ASSETS:
            continue
        s = pd.Series(d["close"], index=d["idx"])
        s = s[(s.index >= de) & (s.index <= ate)]
        if len(s) > 10:
            curvas.append((s / s.iloc[0]).rename(sym))
    if not curvas:
        return None
    eq = pd.concat(curvas, axis=1).ffill().dropna().mean(axis=1) * capital
    dias = max((eq.index[-1] - eq.index[0]).days, 1)
    ret = float(eq.iloc[-1] / capital - 1)
    return {
        "nome": "* COMPRAR E SEGURAR (referencia)",
        "retorno": ret,
        "mensal": (1 + ret) ** (1 / (dias / 30.44)) - 1 if ret > -1 else -1.0,
        "dd": float((eq / eq.cummax() - 1).min()),
        "n": 0, "acerto": float("nan"), "final": float(eq.iloc[-1]),
        "taxas": 0.0,
    }


def rodar_rank(dados, de, ate, capital):
    from v8_simulador import simular, metricas

    originais = {k: getattr(v6_ciclo, k) for k in PARAMETROS}
    linhas = []

    ref = comprar_e_segurar(dados, de, ate, capital)
    if ref:
        linhas.append(ref)

    print(f"\nRodando {len(CONFIGS)} configuracoes de {de:%Y-%m-%d} a "
          f"{ate:%Y-%m-%d}.")
    print("Cada uma e um replay minuto a minuto — leva alguns minutos.\n")

    for i, (nome, mud) in enumerate(CONFIGS, 1):
        print(f"  [{i}/{len(CONFIGS)}] {nome} ...", end="", flush=True)
        try:
            _aplicar(mud, originais)
            r = simular(dados, de, ate, capital, silencioso=True)
            m = metricas(r)
            m["nome"] = nome
            m["_ops"] = r["operacoes"]
            linhas.append(m)
            print(f" {m['retorno']*100:+.2f}%  ({m['n']} ops)", flush=True)
        except Exception as e:
            print(f" FALHOU: {e}", flush=True)
        finally:
            _aplicar({}, originais)

    linhas.sort(key=lambda x: x["mensal"], reverse=True)

    print("\n" + "=" * 100)
    print(f"  RANKING — {de:%Y-%m-%d} a {ate:%Y-%m-%d}  "
          f"(capital inicial ${capital:,.0f})")
    print("=" * 100)
    print(f"\n  {'#':<3} {'Configuracao':<30} {'ao mes':>9} {'periodo':>10} "
          f"{'pior queda':>11} {'ops':>6} {'acerto':>8} {'final':>12}")
    print("  " + "-" * 96)
    for pos, m in enumerate(linhas, 1):
        acerto = ("     —" if np.isnan(m["acerto"])
                  else f"{m['acerto']*100:>7.1f}%")
        print(f"  {pos:<3} {m['nome']:<30} {m['mensal']*100:>+8.2f}% "
              f"{m['retorno']*100:>+9.2f}% {m['dd']*100:>10.1f}% "
              f"{m['n']:>6} {acerto} ${m['final']:>11,.2f}")

    print("\n  " + "-" * 96)
    positivas = [m for m in linhas if m["mensal"] > 0 and m["n"] > 0]
    print(f"  Configuracoes com retorno positivo: {len(positivas)} de "
          f"{len([m for m in linhas if m['n'] > 0])}")
    if ref:
        bateram = [m for m in linhas
                   if m["n"] > 0 and m["retorno"] > ref["retorno"]]
        print(f"  Que bateram comprar-e-segurar: {len(bateram)}")

    print("\n  COMO LER")
    print("    'ao mes' e o retorno do periodo convertido para mes — NAO e")
    print("    garantia de ganhar isso todo mes. 'pior queda' e quanto a conta")
    print("    encolheu do topo ate o fundo: e a dor que voce teria aguentado.")
    print("    Menos de ~50 operacoes: trate o resultado como sorte, nao skill.")
    print("=" * 100)

    return linhas
