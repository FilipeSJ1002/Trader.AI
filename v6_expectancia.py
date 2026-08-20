# -*- coding: utf-8 -*-
"""
v6_expectancia.py — Trader.AI: qual faixa de confianca paga o pedagio?
=======================================================================

A pergunta que decide o rumo depois de 11/08/2026.

O executor rodou 7 dias na testnet: lucro BRUTO de +$26,73 e taxas de −$28,17.
A estrategia tem edge — o edge e menor que o custo de operar. Com TP 1,0% /
SL 0,5% e ~35% de acerto, a expectativa bruta por operacao e ~+0,025% do
nocional contra 0,08% de taxa (0,04% x 2 lados). Perde por construcao.

Esta ferramenta mede, com poder estatistico de verdade, a UNICA coisa que
resolve isso: existe alguma faixa de dir_conf em que a expectancia LIQUIDA
(depois da taxa) e positiva?

Diferenca para as ferramentas anteriores:

  v6_edge_por_faixa.py  mede o retorno no HORIZONTE (120 min a frente).
                        Uma janela pode terminar +0,3% e ter tocado o stop no
                        caminho — nesse caso o trade perdeu, e o edge medido
                        mente.

  v6_expectancia.py     simula a CORRIDA TP x SL minuto a minuto, com a mesma
                        logica do v5_backtest (stop tem prioridade quando os
                        dois disparam no mesmo candle) e desconta a taxa.
                        O numero que sai e o que o dinheiro faz.

Metodologia (Protocolo para IAs): tudo medido na VALIDACAO. O split de teste
so pode ser tocado uma vez, na confirmacao final de uma decisao ja tomada.

Simplificacao assumida e declarada: a saida por SINAL CONTRARIO nao e
simulada (no backtest ela respondeu por 3 de 55 saidas, ~5%). Saidas por TP,
SL e TEMPO — 95% dos casos — sao fieis ao backtest.

Uso:
  python v6_expectancia.py                      # validacao (H2-2025), com gatilho V1
  python v6_expectancia.py --sem-v1             # todas as janelas, sem o gatilho
  python v6_expectancia.py --from 2026-01-01 --to 2026-05-31   # CUIDADO: teste
"""
import os
import sys
import argparse
import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from v5_model import load_model
from v5_data_prep import ASSETS, BTC, WINDOW_SIZE, _load_parquet, _add_features
from v5_backtest import v1_scores, FEE

FAIXAS = [(0.50, 0.52, "abaixo do limiar"),
          (0.52, 0.57, "1x"),
          (0.57, 0.62, "2x"),
          (0.62, 0.67, "5x (a faixa 'boa')"),
          (0.67, 0.72, "10x desativada"),
          (0.72, 1.01, "20x desativada")]


def simular_saida(direcao, entry, hi, lo, close, i, sl_pct, tp_pct, max_hold):
    """
    Corrida TP x SL a partir do candle i+1, ate max_hold minutos.
    Mesma convencao do v5_backtest: o STOP tem prioridade quando os dois
    disparam no mesmo minuto (conservador).

    Devolve (retorno_bruto_fracao, causa).
    """
    fim = min(i + 1 + max_hold, len(close))
    if fim <= i + 1:
        return None, None
    seg_hi, seg_lo = hi[i + 1:fim], lo[i + 1:fim]

    if direcao == "LONG":
        sl_price, tp_price = entry * (1 - sl_pct), entry * (1 + tp_pct)
        toca_sl = np.nonzero(seg_lo <= sl_price)[0]
        toca_tp = np.nonzero(seg_hi >= tp_price)[0]
    else:
        sl_price, tp_price = entry * (1 + sl_pct), entry * (1 - tp_pct)
        toca_sl = np.nonzero(seg_hi >= sl_price)[0]
        toca_tp = np.nonzero(seg_lo <= tp_price)[0]

    p_sl = int(toca_sl[0]) if len(toca_sl) else None
    p_tp = int(toca_tp[0]) if len(toca_tp) else None

    if p_sl is not None and (p_tp is None or p_sl <= p_tp):
        saida, causa = sl_price, "SL"
    elif p_tp is not None:
        saida, causa = tp_price, "TP"
    else:
        saida, causa = close[fim - 1], "TEMPO"

    bruto = (saida / entry - 1) if direcao == "LONG" else (1 - saida / entry)
    return bruto, causa


def main():
    ap = argparse.ArgumentParser(description="Expectancia liquida por faixa de dir_conf")
    ap.add_argument("--model", default="v5_model_b.pth")
    ap.add_argument("--from", dest="dt_from", default="2025-07-01",
                    help="padrao: inicio da VALIDACAO")
    ap.add_argument("--to", dest="dt_to", default="2025-12-31",
                    help="padrao: fim da VALIDACAO")
    ap.add_argument("--step", type=int, default=15, help="minutos entre avaliacoes")
    ap.add_argument("--sl", type=float, default=0.005)
    ap.add_argument("--tp", type=float, default=0.010)
    ap.add_argument("--max-hold", dest="max_hold", type=int, default=360)
    ap.add_argument("--v1-thresh", dest="v1_thresh", type=int, default=60)
    ap.add_argument("--sem-v1", dest="sem_v1", action="store_true",
                    help="ignora o gatilho V1 e o regime (mede a populacao inteira)")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"
    taxa = FEE * 2          # ida e volta, fracao do nocional

    btc = _load_parquet(BTC)
    model = None
    confs, retornos, causas, ativos_lin = [], [], [], []
    p_abs, p_neutros, forcas = [], [], []

    print(f"\nSimulando {'TODAS as janelas' if a.sem_v1 else 'candidatos do V1'} "
          f"({a.dt_from} -> {a.dt_to})...")

    for sym in ASSETS:
        adf = _load_parquet(sym)
        com = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[com], btc.loc[com]
        fdf = _add_features(a2, b2).dropna()
        if len(fdf) < WINDOW_SIZE + 10:
            continue

        X = fdf.values.astype(np.float32)
        cols = list(fdf.columns)
        close = a2["close"].reindex(fdf.index).values.astype(np.float64)
        hi = a2["high"].reindex(fdf.index).values.astype(np.float64)
        lo = a2["low"].reindex(fdf.index).values.astype(np.float64)
        sma24_s = a2["close"].rolling(1440).mean().reindex(fdf.index)
        sma24 = sma24_s.values
        # FORCA da tendencia — mesma formula do v5_backtest.precompute():
        # |preco - SMA24h| / SMA24h normalizada pela volatilidade tipica do ativo
        close_s = a2["close"].reindex(fdf.index)
        dist_rel = ((close_s - sma24_s).abs() / (sma24_s + 1e-9))
        vol_tipica = dist_rel.rolling(1440, min_periods=120).median()
        forca_arr = (dist_rel / (vol_tipica + 1e-9)).fillna(0.0).values

        if model is None:
            model = load_model(a.model, X.shape[1], device)

        janela = (fdf.index >= a.dt_from) & (fdf.index <= a.dt_to)
        idxs = np.where(janela)[0]
        idxs = idxs[(idxs >= WINDOW_SIZE) & (idxs < len(close) - 2)][::a.step]

        # Portao 1+2: gatilho tecnico do V1 + regime diario (identico ao sistema)
        candidatos = []
        for i in idxs:
            regime_down = bool(close[i] < sma24[i]) if not np.isnan(sma24[i]) else False
            if a.sem_v1:
                candidatos.append((i, "SHORT" if regime_down else "LONG"))
                continue
            b_sc, s_sc, _ = v1_scores(X[i], X[i - 1], cols)
            if b_sc >= a.v1_thresh and not regime_down:
                candidatos.append((i, "LONG"))
            elif s_sc >= a.v1_thresh and regime_down:
                candidatos.append((i, "SHORT"))

        if not candidatos:
            print(f"  {sym}: nenhum candidato")
            continue

        # Portao 3: confianca direcional do modelo
        for i0 in range(0, len(candidatos), 256):
            lote = candidatos[i0:i0 + 256]
            batch = np.stack([X[i - WINDOW_SIZE:i] for i, _ in lote])
            with torch.no_grad():
                logits, _ = model(torch.tensor(batch).to(device))
                p = torch.softmax(logits, 1).cpu().numpy()
            p_dn, p_up = p[:, 0], p[:, 2]
            p_dir = p_up + p_dn + 1e-9

            for k, (i, direcao) in enumerate(lote):
                dc = float((p_up[k] if direcao == "LONG" else p_dn[k]) / p_dir[k])
                bruto, causa = simular_saida(direcao, close[i], hi, lo, close,
                                             i, a.sl, a.tp, a.max_hold)
                if bruto is None:
                    continue
                confs.append(dc)
                retornos.append(bruto - taxa)      # LIQUIDO de taxa
                causas.append(causa)
                ativos_lin.append(sym)
                # p_dir ABSOLUTO (nao a razao): a hipotese e que dir_conf alto
                # com p_alta/p_queda minusculos significa "nada vai acontecer"
                p_abs.append(float(p_up[k] if direcao == "LONG" else p_dn[k]))
                p_neutros.append(float(p[k, 1]))
                forcas.append(float(forca_arr[i]))

        print(f"  {sym}: {len(candidatos)} candidatos", flush=True)
        del adf, a2, b2, fdf, X

    if not confs:
        print("Nenhum candidato no periodo.")
        return

    conf = np.array(confs)
    ret = np.array(retornos)
    causa = np.array(causas)

    print(f"\n{'='*82}")
    print(f"  EXPECTANCIA LIQUIDA POR FAIXA — {a.model}")
    print(f"  {a.dt_from} -> {a.dt_to} | TP {a.tp*100:.1f}% / SL {a.sl*100:.1f}% "
          f"| taxa {taxa*100:.3f}% ida+volta")
    print(f"  Gatilho: {'NENHUM (todas as janelas)' if a.sem_v1 else f'V1 >= {a.v1_thresh} + regime'}")
    print(f"{'='*82}")
    print(f"  {'Faixa':>11} {'rotulo':>20} {'ops':>7} {'%TP':>6} {'%SL':>6} "
          f"{'%tempo':>7} {'EXPECT.':>9} {'total':>10}")
    print("  " + "-" * 80)

    for lo_f, hi_f, rot in FAIXAS:
        m = (conf >= lo_f) & (conf < hi_f)
        n = int(m.sum())
        if n == 0:
            print(f"  {lo_f:.2f}-{hi_f:.2f} {rot:>20} {n:>7}")
            continue
        r, c = ret[m], causa[m]
        exp = r.mean()
        print(f"  {lo_f:.2f}-{hi_f:.2f} {rot:>20} {n:>7} "
              f"{(c=='TP').mean()*100:>5.1f}% {(c=='SL').mean()*100:>5.1f}% "
              f"{(c=='TEMPO').mean()*100:>6.1f}% "
              f"{exp*100:>+8.4f}% {r.sum()*100:>+9.1f}%")

    print("\n  Acumulado — operar SO acima do limiar (e o que muda o threshold):")
    print(f"  {'limiar':>8} {'ops':>8} {'% do total':>11} {'%TP':>6} "
          f"{'EXPECT. liq':>12} {'bruto':>10}")
    print("  " + "-" * 62)
    for lim in [0.50, 0.52, 0.57, 0.62, 0.67, 0.72]:
        m = conf >= lim
        n = int(m.sum())
        if n < 30:
            print(f"  {lim:>8.2f} {n:>8} {'(poucas)':>11}")
            continue
        r, c = ret[m], causa[m]
        print(f"  {lim:>8.2f} {n:>8} {n/len(conf)*100:>10.1f}% "
              f"{(c=='TP').mean()*100:>5.1f}% {r.mean()*100:>+11.4f}% "
              f"{(r.mean()+taxa)*100:>+9.4f}%")

    # ── A hipotese: dir_conf alto = "nada vai acontecer", nao "vai subir" ───
    pa = np.array(p_abs)
    pn = np.array(p_neutros)
    print(f"\n  DIAGNOSTICO — o que 'confianca alta' significa de fato:")
    print(f"  {'Faixa dir_conf':>16} {'ops':>7} {'p_neutro medio':>16} "
          f"{'p_direcao medio':>16}")
    print("  " + "-" * 60)
    for lo_f, hi_f, _rot in FAIXAS:
        m = (conf >= lo_f) & (conf < hi_f)
        if not m.any():
            continue
        print(f"  {lo_f:.2f}-{hi_f:.2f}{'':>6} {int(m.sum()):>7} "
              f"{pn[m].mean():>15.3f} {pa[m].mean():>16.3f}")

    # ── Expectancia por probabilidade ABSOLUTA da direcao ───────────────────
    print(f"\n  EXPECTANCIA POR p_direcao ABSOLUTO (em vez da razao dir_conf):")
    print(f"  {'p_direcao >=':>13} {'ops':>8} {'% do total':>11} {'%TP':>6} "
          f"{'EXPECT. liq':>12}")
    print("  " + "-" * 56)
    for lim in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        m = pa >= lim
        n = int(m.sum())
        if n < 30:
            print(f"  {lim:>13.2f} {n:>8} {'(poucas)':>11}")
            continue
        r, c = ret[m], causa[m]
        print(f"  {lim:>13.2f} {n:>8} {n/len(conf)*100:>10.1f}% "
              f"{(c=='TP').mean()*100:>5.1f}% {r.mean()*100:>+11.4f}%")

    # ── p_movimento = 1 - p_neutro: "vai acontecer alguma coisa?" ───────────
    # Esta e a metrica candidata a substituir dir_conf na curva de alavancagem.
    p_mov = 1.0 - pn
    print(f"\n  EXPECTANCIA POR p_movimento (1 - p_neutro) — a metrica candidata:")
    print(f"  {'Faixa p_mov':>13} {'ops':>7} {'%TP':>6} {'%SL':>6} {'%tempo':>7} "
          f"{'EXPECT. liq':>12} {'bruto':>9}")
    print("  " + "-" * 66)
    bandas_mov = [(0.00, 0.40), (0.40, 0.50), (0.50, 0.60),
                  (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    for lo_f, hi_f in bandas_mov:
        m = (p_mov >= lo_f) & (p_mov < hi_f)
        n = int(m.sum())
        if n == 0:
            continue
        r, c = ret[m], causa[m]
        print(f"  {lo_f:.2f}-{hi_f:.2f}{'':>3} {n:>7} "
              f"{(c=='TP').mean()*100:>5.1f}% {(c=='SL').mean()*100:>5.1f}% "
              f"{(c=='TEMPO').mean()*100:>6.1f}% {r.mean()*100:>+11.4f}% "
              f"{(r.mean()+taxa)*100:>+8.4f}%")

    print(f"\n  Acumulado por p_movimento (operar SO acima do limiar):")
    print(f"  {'limiar':>8} {'ops':>8} {'% do total':>11} {'%TP':>6} "
          f"{'EXPECT. liq':>12} {'bruto':>9}")
    print("  " + "-" * 60)
    for lim in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        m = p_mov >= lim
        n = int(m.sum())
        if n < 30:
            print(f"  {lim:>8.2f} {n:>8} {'(poucas)':>11}")
            continue
        r, c = ret[m], causa[m]
        print(f"  {lim:>8.2f} {n:>8} {n/len(conf)*100:>10.1f}% "
              f"{(c=='TP').mean()*100:>5.1f}% {r.mean()*100:>+11.4f}% "
              f"{(r.mean()+taxa)*100:>+8.4f}%")

    # ── A PERGUNTA CENTRAL: o edge so existe quando ha tendencia? ──────────
    fo = np.array(forcas)
    print(f"\n  EXPECTANCIA POR FORCA DA TENDENCIA — operar so quando ha tendencia?")
    print(f"  {'Faixa forca':>13} {'ops':>7} {'%TP':>6} {'%SL':>6} {'%tempo':>7} "
          f"{'EXPECT. liq':>12} {'bruto':>9}")
    print("  " + "-" * 66)
    for lo_f, hi_f in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                       (2.0, 3.0), (3.0, 99.0)]:
        m = (fo >= lo_f) & (fo < hi_f)
        n = int(m.sum())
        if n == 0:
            continue
        r, c = ret[m], causa[m]
        print(f"  {lo_f:>5.1f}-{hi_f:<5.1f}{'':>2} {n:>7} "
              f"{(c=='TP').mean()*100:>5.1f}% {(c=='SL').mean()*100:>5.1f}% "
              f"{(c=='TEMPO').mean()*100:>6.1f}% {r.mean()*100:>+11.4f}% "
              f"{(r.mean()+taxa)*100:>+8.4f}%")

    print(f"\n  Acumulado — PORTAO DE ENTRADA por forca (nao operar abaixo do limiar):")
    print(f"  {'forca >=':>9} {'ops':>8} {'% mantido':>11} {'%TP':>6} "
          f"{'EXPECT. liq':>12} {'erro-padrao':>13}")
    print("  " + "-" * 64)
    for lim in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        m = fo >= lim
        n = int(m.sum())
        if n < 30:
            print(f"  {lim:>9.1f} {n:>8} {'(poucas)':>11}")
            continue
        r, c = ret[m], causa[m]
        se = r.std(ddof=1) / np.sqrt(n)
        print(f"  {lim:>9.1f} {n:>8} {n/len(conf)*100:>10.1f}% "
              f"{(c=='TP').mean()*100:>5.1f}% {r.mean()*100:>+11.4f}% "
              f"{'+/- ' + format(se*100, '.4f') + '%':>13}")

    print(f"\n{'='*82}")
    print("  LEITURA")
    print("    EXPECT. liq = ganho medio por operacao, em % do nocional, JA com taxa.")
    print("    Positivo em alguma faixa -> existe threshold que paga o pedagio.")
    print("    Negativo em todas       -> nenhum ajuste de limiar salva; o problema")
    print("                               e o poder preditivo, nao a selecao.")
    print(f"    Para referencia: a taxa sozinha custa {taxa*100:.3f}% por operacao.")
    print(f"{'='*82}")


if __name__ == "__main__":
    main()
