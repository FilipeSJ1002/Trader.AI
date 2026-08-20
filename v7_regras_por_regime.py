# -*- coding: utf-8 -*-
"""
v7_regras_por_regime.py — a logica de regras tem edge, e em qual regime?
=========================================================================

A pergunta que decide o V7.

O V1 (regras: RSI + MACD + Bollinger + confluencia) rendeu +16% num mes de
bull real e perdeu ~25% quando o mercado virou. Duas explicacoes possiveis:

  (a) EDGE REAL em bull  -> vale construir a "logica de alta" em cima dele
  (b) BETA de mercado    -> ele so subiu junto com o BTC, sem vantagem
                            propria; qualquer compra teria feito igual

Distinguir (a) de (b) e o que este script faz. Duas vantagens sobre tudo que
medimos ate agora:

  1. Regras NAO SAO TREINADAS -> nao ha contaminacao de dados. Podemos medir
     em 2019-2026 inteiro (dois ciclos completos), e nao so nos 6 meses de
     validacao. Poder estatistico ~100x maior.
  2. Comparacao contra DOIS baselines honestos:
       - passeio aleatorio  SL/(SL+TP)  -> a regra escolhe melhor que o acaso?
       - comprar e segurar             -> a regra bate o mercado?

Regimes (definidos por SMA de 200 dias sobre candles diarios):
    BULL     preco acima da SMA200 e SMA200 subindo
    BEAR     preco abaixo da SMA200 e SMA200 caindo
    LATERAL  o resto

Uso:
  python v7_regras_por_regime.py                       # V1, historico completo
  python v7_regras_por_regime.py --regra candle       # padroes do TradingView
  python v7_regras_por_regime.py --sl 0.02 --tp 0.04  # stops largos
"""
import sys
import argparse
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from v5_data_prep import ASSETS, BTC, _load_parquet, _add_features
from v5_backtest import v1_scores, FEE


def classifica_regime(close_s):
    """
    BULL / BEAR / LATERAL a partir da SMA200 DIARIA.
    Devolve um array alinhado ao indice de minutos.
    """
    diario = close_s.resample("1D").last()
    sma200 = diario.rolling(200).mean()
    subindo = sma200.diff(20) > 0
    acima = diario > sma200
    reg_d = pd.Series("LATERAL", index=diario.index)
    reg_d[acima & subindo] = "BULL"
    reg_d[(~acima) & (~subindo)] = "BEAR"
    return reg_d.reindex(close_s.index, method="ffill").values


def sinais_v1(X, cols, i, thresh):
    """Gatilho do V1: score de confluencia (mesma funcao do backtest)."""
    b_sc, s_sc, _ = v1_scores(X[i], X[i - 1], cols)
    if b_sc >= thresh:
        return "LONG"
    if s_sc >= thresh:
        return "SHORT"
    return None


def sinais_candle(o, h, l, c, v, vavg, rsi, ma, i, rsi_os=30, rsi_ob=70,
                  vol_mult=1.5):
    """
    Padroes do script do TradingView: martelo / estrela cadente / engolfo,
    com filtros de RSI, media movel e volume.

    Correcao aplicada: o pavio inferior usa min(open, close) - low, e nao
    open - low (o script original soma o corpo ao pavio em candles de baixa).
    """
    faixa = h[i] - l[i]
    if faixa <= 0:
        return None
    corpo = abs(c[i] - o[i])
    pav_inf = min(o[i], c[i]) - l[i]
    pav_sup = h[i] - max(o[i], c[i])

    martelo = (pav_inf > faixa * 0.5 and corpo < faixa * 0.3
               and (h[i] - c[i]) < faixa * 0.25)
    estrela = (pav_sup > faixa * 0.5 and corpo < faixa * 0.3
               and (c[i] - l[i]) < faixa * 0.25)
    engolfo_alta = (c[i-1] < o[i-1] and c[i] > o[i]
                    and c[i] > o[i-1] and o[i] < c[i-1])
    engolfo_baixa = (c[i-1] > o[i-1] and c[i] < o[i]
                     and o[i] > c[i-1] and c[i] < o[i-1])

    passa_vol = v[i] > vavg[i] * vol_mult
    if not passa_vol:
        return None
    if (martelo or engolfo_alta) and rsi[i] < rsi_os and c[i] > ma[i]:
        return "LONG"
    if (estrela or engolfo_baixa) and rsi[i] > rsi_ob and c[i] < ma[i]:
        return "SHORT"
    return None


def simular(direcao, entry, hi, lo, close, i, sl_pct, tp_pct, max_hold):
    """Corrida TP x SL (stop tem prioridade no empate, como no v5_backtest)."""
    fim = min(i + 1 + max_hold, len(close))
    if fim <= i + 1:
        return None, None
    seg_hi, seg_lo = hi[i + 1:fim], lo[i + 1:fim]
    if direcao == "LONG":
        sl_p, tp_p = entry * (1 - sl_pct), entry * (1 + tp_pct)
        t_sl = np.nonzero(seg_lo <= sl_p)[0]
        t_tp = np.nonzero(seg_hi >= tp_p)[0]
    else:
        sl_p, tp_p = entry * (1 + sl_pct), entry * (1 - tp_pct)
        t_sl = np.nonzero(seg_hi >= sl_p)[0]
        t_tp = np.nonzero(seg_lo <= tp_p)[0]
    p_sl = int(t_sl[0]) if len(t_sl) else None
    p_tp = int(t_tp[0]) if len(t_tp) else None
    if p_sl is not None and (p_tp is None or p_sl <= p_tp):
        saida, causa = sl_p, "SL"
    elif p_tp is not None:
        saida, causa = tp_p, "TP"
    else:
        saida, causa = close[fim - 1], "TEMPO"
    bruto = (saida / entry - 1) if direcao == "LONG" else (1 - saida / entry)
    return bruto, causa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regra", choices=["v1", "candle"], default="v1")
    ap.add_argument("--from", dest="dt_from", default="2019-01-01")
    ap.add_argument("--to", dest="dt_to", default="2026-12-31")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--sl", type=float, default=0.005)
    ap.add_argument("--tp", type=float, default=0.010)
    ap.add_argument("--max-hold", dest="max_hold", type=int, default=360)
    ap.add_argument("--v1-thresh", dest="v1_thresh", type=int, default=60)
    ap.add_argument("--assets", default="6",
                    help="'6' (os do treino) ou 'all' (todos os parquets)")
    ap.add_argument("--detalhe", default=None,
                    help="Recorte para abrir por ativo e ano, ex: BULL/LONG")
    a = ap.parse_args()

    taxa = FEE * 2
    btc = _load_parquet(BTC)
    linhas = []

    if a.assets.lower() == "all":
        import glob, os
        universo = sorted(os.path.basename(p).replace("_1m.parquet", "")
                          for p in glob.glob("data/*_1m.parquet"))
    else:
        universo = list(ASSETS)
    print(f"  Universo: {len(universo)} ativos")

    print(f"\nMedindo regra '{a.regra}' em {a.dt_from} -> {a.dt_to} "
          f"| TP {a.tp*100:.1f}% SL {a.sl*100:.1f}% hold {a.max_hold}min")

    for sym in universo:
        adf = _load_parquet(sym)
        com = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[com], btc.loc[com]

        if a.regra == "v1":
            fdf = _add_features(a2, b2).dropna()
            X = fdf.values.astype(np.float32)
            cols = list(fdf.columns)
            idx_ref = fdf.index
        else:
            idx_ref = a2.index
            X, cols = None, None

        c = a2["close"].reindex(idx_ref).values.astype(np.float64)
        h = a2["high"].reindex(idx_ref).values.astype(np.float64)
        l = a2["low"].reindex(idx_ref).values.astype(np.float64)
        o = a2["open"].reindex(idx_ref).values.astype(np.float64)
        v = a2["volume"].reindex(idx_ref).values.astype(np.float64)
        close_s = a2["close"].reindex(idx_ref)
        regimes = classifica_regime(close_s)

        if a.regra == "candle":
            vavg = close_s.rolling(20).mean().values * 0 + \
                   pd.Series(v, index=idx_ref).rolling(20).mean().values
            delta = close_s.diff()
            ganho = delta.clip(lower=0).rolling(14).mean()
            perda = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = (100 - 100 / (1 + ganho / (perda + 1e-12))).values
            ma = close_s.rolling(200).mean().values

        jan = (idx_ref >= a.dt_from) & (idx_ref <= a.dt_to)
        ini = 200 if a.regra == "v1" else 220
        idxs = np.where(jan)[0]
        idxs = idxs[(idxs >= ini) & (idxs < len(c) - a.max_hold - 2)][::a.step]

        n_sym = 0
        for i in idxs:
            if a.regra == "v1":
                direcao = sinais_v1(X, cols, i, a.v1_thresh)
            else:
                direcao = sinais_candle(o, h, l, c, v, vavg, rsi, ma, i)
            if direcao is None:
                continue
            bruto, causa = simular(direcao, c[i], h, l, c, i,
                                   a.sl, a.tp, a.max_hold)
            if bruto is None:
                continue
            linhas.append((regimes[i], direcao, bruto - taxa, causa,
                           sym, idx_ref[i].year))
            n_sym += 1

        print(f"  {sym}: {n_sym} sinais", flush=True)
        del adf, a2, b2

    if not linhas:
        print("Nenhum sinal no periodo.")
        return

    reg = np.array([x[0] for x in linhas])
    dirs = np.array([x[1] for x in linhas])
    ret = np.array([x[2] for x in linhas], dtype=float)
    causa = np.array([x[3] for x in linhas])

    acaso = a.sl / (a.sl + a.tp)      # baseline do passeio aleatorio

    print(f"\n{'='*88}")
    print(f"  REGRA '{a.regra.upper()}' POR REGIME DE MERCADO — {a.dt_from} a {a.dt_to}")
    print(f"  TP {a.tp*100:.1f}% / SL {a.sl*100:.1f}% | taxa {taxa*100:.3f}% "
          f"| acerto esperado por acaso: {acaso*100:.1f}%")
    print(f"{'='*88}")
    print(f"  {'Regime':<9} {'Direcao':<7} {'ops':>7} {'%TP':>7} {'vs acaso':>9} "
          f"{'EXPECT. liq':>12} {'erro-padrao':>13} {'t':>6}")
    print("  " + "-" * 86)

    for r in ["BULL", "LATERAL", "BEAR"]:
        for d in ["LONG", "SHORT"]:
            m = (reg == r) & (dirs == d)
            n = int(m.sum())
            if n < 30:
                continue
            rr, cc = ret[m], causa[m]
            tp_rate = (cc == "TP").mean()
            se = rr.std(ddof=1) / np.sqrt(n)
            t = rr.mean() / se if se > 0 else 0.0
            print(f"  {r:<9} {d:<7} {n:>7} {tp_rate*100:>6.1f}% "
                  f"{(tp_rate-acaso)*100:>+8.1f}pp {rr.mean()*100:>+11.4f}% "
                  f"{'+/- ' + format(se*100, '.4f') + '%':>13} {t:>+6.2f}")
        m = (reg == r)
        if m.sum() >= 30:
            rr = ret[m]
            se = rr.std(ddof=1) / np.sqrt(len(rr))
            print(f"  {r:<9} {'TOTAL':<7} {int(m.sum()):>7} {'':>7} {'':>9} "
                  f"{rr.mean()*100:>+11.4f}% "
                  f"{'+/- ' + format(se*100, '.4f') + '%':>13} "
                  f"{rr.mean()/se if se>0 else 0:>+6.2f}")
        print()

    se_g = ret.std(ddof=1) / np.sqrt(len(ret))
    print(f"  GERAL: {len(ret)} operacoes | expectancia {ret.mean()*100:+.4f}% "
          f"+/- {se_g*100:.4f}% | t = {ret.mean()/se_g:+.2f}")

    # ── Robustez: um edge real aparece em VARIOS ativos e VARIOS anos ───────
    if a.detalhe:
        r_alvo, d_alvo = a.detalhe.upper().split("/")
        syms = np.array([x[4] for x in linhas])
        anos = np.array([x[5] for x in linhas])
        m0 = (reg == r_alvo) & (dirs == d_alvo)
        print(f"\n{'='*88}")
        print(f"  ROBUSTEZ DO RECORTE {r_alvo}/{d_alvo} — n = {int(m0.sum())}")
        print("  Um edge real se repete entre ativos e entre anos. Se estiver")
        print("  concentrado em um so, e sorte com cara de padrao.")
        print(f"{'='*88}")

        for titulo, chaves, vetor in (("POR ATIVO", sorted(set(syms[m0])), syms),
                                      ("POR ANO", sorted(set(anos[m0])), anos)):
            print(f"\n  {titulo}")
            print(f"  {'':<12} {'ops':>6} {'%TP':>7} {'EXPECT. liq':>12} "
                  f"{'erro-padrao':>13} {'t':>6}")
            print("  " + "-" * 60)
            positivos = 0
            for ch in chaves:
                m = m0 & (vetor == ch)
                n = int(m.sum())
                if n < 15:
                    print(f"  {str(ch):<12} {n:>6}  (amostra pequena)")
                    continue
                rr, cc = ret[m], causa[m]
                se = rr.std(ddof=1) / np.sqrt(n)
                t = rr.mean() / se if se > 0 else 0.0
                if rr.mean() > 0:
                    positivos += 1
                print(f"  {str(ch):<12} {n:>6} {(cc=='TP').mean()*100:>6.1f}% "
                      f"{rr.mean()*100:>+11.4f}% "
                      f"{'+/- ' + format(se*100, '.4f') + '%':>13} {t:>+6.2f}")
            print(f"  -> {positivos} de {len(chaves)} com expectancia positiva")
    print(f"\n{'='*88}")
    print("  COMO LER")
    print("    'vs acaso' positivo = a regra escolhe melhor que um passeio aleatorio.")
    print("    |t| >= 2 = o resultado nao e ruido.")
    print(f"    Para valer a pena, a expectancia precisa ser positiva DEPOIS da taxa.")
    print(f"{'='*88}")


if __name__ == "__main__":
    main()
