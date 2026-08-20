# -*- coding: utf-8 -*-
"""
v7_dataset.py — a base supervisionada do V7
============================================

Gera UM conjunto de janelas e, sobre elas, TODAS as rotulagens e horizontes
que vamos comparar. Assim os modelos disputam em igualdade: mesma amostra,
mesmas features, muda so o que se pede para prever.

TRES ROTULAGENS
---------------
  tb    TRIPLE BARRIER (Lopez de Prado) — qual barreira foi tocada PRIMEIRO:
        0 = stop | 1 = tempo esgotou | 2 = alvo
        Corrige o defeito estrutural do projeto: rotular por variacao em
        horizonte fixo ignora o CAMINHO do preco. Uma janela que termina
        +0,9% mas caiu -0,5% no meio era rotulada ALTA — e na operacao real
        teria sido estopada. O modelo aprendia uma coisa, o sistema fazia outra.

  meta  META-LABELING — binario: a operacao que a regra disparou teria dado
        certo? 1 = sim (bateu o alvo) | 0 = nao.
        Nao pede para prever o mercado, so para julgar um sinal ja existente.
        Pergunta mais facil, e a probabilidade que sai serve de tamanho da posicao.

  dir   DIRECAO EM HORIZONTE FIXO — o esquema atual do projeto, mantido para
        comparacao: 0 = queda | 1 = neutro | 2 = alta, pela variacao no fim
        do horizonte.

QUATRO CONFIGURACOES DE ALVO
----------------------------
  A  alvo 1,0% / stop 0,5% / 6h      (a de producao hoje)
  B  alvo 3,0% / stop 1,5% / 2 dias  (taxa cai de 8% para 2,7% do alvo)
  C  ATR adaptativo, alvo = 2x stop  (stop fora do ruido de cada ativo)
  D  alvo 0,5% / stop 0,5% / 1h      (onde a medicao de 15/08 achou sinal)

O QUE VAI PARA O DISCO
----------------------
  X_seq   (n, 120, 18)  janelas — para modelos de sequencia (BiLSTM, MiniRocket)
  X_tab   (n, ~90)      features tabulares — para arvores e boosting
  y_*     rotulos de cada esquema x configuracao
  pnl_*   RESULTADO FINANCEIRO liquido de cada operacao, ja calculado

O pnl_* e o que permite avaliar um modelo por DINHEIRO e nao so por acuracia:
basta somar o pnl das operacoes que ele mandaria fazer.

Uso:
  python v7_dataset.py                    # 11 ativos, historico completo
  python v7_dataset.py --assets 6 --step 15
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from v5_data_prep import ASSETS, BTC, WINDOW_SIZE, _load_parquet, _add_features
from v5_backtest import v1_scores, FEE

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_v7")
os.makedirs(DEST, exist_ok=True)

# nome -> (tp, sl, max_hold_min, modo)   modo: "fixo" ou "atr"
CONFIGS = {
    "A_1x0.5_6h":  (0.010, 0.005, 360,  "fixo"),
    "B_3x1.5_2d":  (0.030, 0.015, 2880, "fixo"),
    "C_atr_6h":    (None,  None,  360,  "atr"),
    "D_0.5x0.5_1h": (0.005, 0.005, 60,  "fixo"),
}
ATR_K, ATR_PISO, ATR_TETO = 1.2, 0.004, 0.020


def simula(direcao, entry, hi, lo, close, i, sl_pct, tp_pct, max_hold):
    """
    Corrida alvo x stop, minuto a minuto, a partir de i+1.
    Convencao identica ao v5_backtest: o STOP ganha quando os dois disparam
    no mesmo minuto (conservador).

    Devolve (rotulo_tb, retorno_bruto).
      rotulo_tb: 0 = stop | 1 = tempo | 2 = alvo
    """
    fim = min(i + 1 + max_hold, len(close))
    if fim <= i + 1:
        return None, None
    seg_hi, seg_lo = hi[i + 1:fim], lo[i + 1:fim]

    if direcao == 1:      # comprado
        sl_p, tp_p = entry * (1 - sl_pct), entry * (1 + tp_pct)
        t_sl = np.nonzero(seg_lo <= sl_p)[0]
        t_tp = np.nonzero(seg_hi >= tp_p)[0]
    else:                 # vendido
        sl_p, tp_p = entry * (1 + sl_pct), entry * (1 - tp_pct)
        t_sl = np.nonzero(seg_hi >= sl_p)[0]
        t_tp = np.nonzero(seg_lo <= tp_p)[0]

    p_sl = int(t_sl[0]) if len(t_sl) else None
    p_tp = int(t_tp[0]) if len(t_tp) else None

    if p_sl is not None and (p_tp is None or p_sl <= p_tp):
        saida, rot = sl_p, 0
    elif p_tp is not None:
        saida, rot = tp_p, 2
    else:
        saida, rot = close[fim - 1], 1

    bruto = (saida / entry - 1) if direcao == 1 else (1 - saida / entry)
    return rot, bruto


def features_tabulares(janela, cols):
    """
    Achata a janela (120 x 18) em um vetor para modelos tabulares:
    ultimo valor de cada feature + media, desvio, minimo, maximo e
    variacao (fim - inicio) ao longo da janela.
    """
    ult = janela[-1]
    med = janela.mean(axis=0)
    dp = janela.std(axis=0)
    mn = janela.min(axis=0)
    mx = janela.max(axis=0)
    delta = janela[-1] - janela[0]
    vetor = np.concatenate([ult, med, dp, mn, mx, delta])
    nomes = ([f"{c}_ult" for c in cols] + [f"{c}_med" for c in cols]
             + [f"{c}_dp" for c in cols] + [f"{c}_min" for c in cols]
             + [f"{c}_max" for c in cols] + [f"{c}_delta" for c in cols])
    return vetor.astype(np.float32), nomes


def main():
    ap = argparse.ArgumentParser(description="Gera a base supervisionada do V7")
    ap.add_argument("--assets", default="all", help="'all', '6' ou lista por virgula")
    ap.add_argument("--step", type=int, default=15,
                    help="minutos entre avaliacoes (padrao 15, igual ao backtest)")
    ap.add_argument("--v1-thresh", dest="v1_thresh", type=int, default=60)
    ap.add_argument("--from", dest="dt_from", default="2019-01-01")
    ap.add_argument("--to", dest="dt_to", default="2026-12-31")
    a = ap.parse_args()

    if a.assets == "all":
        universo = sorted(os.path.basename(p).replace("_1m.parquet", "")
                          for p in glob.glob("data/*_1m.parquet"))
    elif a.assets == "6":
        universo = list(ASSETS)
    else:
        universo = [s.strip().upper() for s in a.assets.split(",")]

    taxa = FEE * 2
    btc = _load_parquet(BTC)

    seqs, tabs, metas = [], [], []
    resultados = {k: {"rot": [], "pnl": [], "dir": []} for k in CONFIGS}
    nomes_tab = None
    cols_ref = None

    print(f"\nGerando base V7 | {len(universo)} ativos | {a.dt_from} -> {a.dt_to}")
    print(f"Amostragem: eventos do gatilho V1 >= {a.v1_thresh}, a cada {a.step} min\n")

    for sym in universo:
        adf = _load_parquet(sym)
        com = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[com], btc.loc[com]
        fdf = _add_features(a2, b2).dropna()
        if len(fdf) < WINDOW_SIZE + 3000:
            print(f"  {sym}: historico curto, pulando")
            continue

        X = fdf.values.astype(np.float32)
        cols = list(fdf.columns)
        cols_ref = cols_ref or cols
        c = a2["close"].reindex(fdf.index).values.astype(np.float64)
        h = a2["high"].reindex(fdf.index).values.astype(np.float64)
        l = a2["low"].reindex(fdf.index).values.astype(np.float64)
        sma24 = a2["close"].rolling(1440).mean().reindex(fdf.index).values
        i_atr = cols.index("atr_pct") if "atr_pct" in cols else -1

        jan = (fdf.index >= a.dt_from) & (fdf.index <= a.dt_to)
        limite = max(cfg[2] for cfg in CONFIGS.values())
        idxs = np.where(jan)[0]
        idxs = idxs[(idxs >= WINDOW_SIZE) & (idxs < len(c) - limite - 2)][::a.step]

        n_sym = 0
        for i in idxs:
            regime_down = bool(c[i] < sma24[i]) if not np.isnan(sma24[i]) else False
            b_sc, s_sc, _ = v1_scores(X[i], X[i - 1], cols)
            # Portao 1+2 identico ao sistema: gatilho tecnico + regime diario
            if b_sc >= a.v1_thresh and not regime_down:
                direcao, score = 1, b_sc
            elif s_sc >= a.v1_thresh and regime_down:
                direcao, score = -1, s_sc
            else:
                continue

            # Simula as 4 configuracoes sobre o MESMO evento
            linha_rot, linha_pnl, linha_dir, valido = {}, {}, {}, True
            for nome, (tp, sl, hold, modo) in CONFIGS.items():
                if modo == "atr":
                    atr = float(X[i][i_atr]) if i_atr >= 0 else 0.0
                    sl_i = min(max(ATR_K * atr, ATR_PISO), ATR_TETO) if atr > 0 else 0.005
                    tp_i = sl_i * 2.0
                else:
                    sl_i, tp_i = sl, tp
                rot, bruto = simula(direcao, c[i], h, l, c, i, sl_i, tp_i, hold)
                if rot is None:
                    valido = False
                    break
                linha_rot[nome] = rot
                linha_pnl[nome] = bruto - taxa        # LIQUIDO de taxa

                # DIRECAO em horizonte fixo (esquema atual do projeto): olha so
                # a variacao do ATIVO no fim do horizonte, ignorando o caminho.
                # Limiares acompanham a configuracao, para comparacao justa.
                fim_h = min(i + hold, len(c) - 1)
                ret_fwd = c[fim_h] / c[i] - 1
                linha_dir[nome] = (2 if ret_fwd >= tp_i
                                   else 0 if ret_fwd <= -tp_i else 1)
            if not valido:
                continue

            janela = X[i - WINDOW_SIZE:i]
            vetor, nomes = features_tabulares(janela, cols)
            nomes_tab = nomes_tab or nomes

            seqs.append(janela)
            tabs.append(vetor)
            metas.append({
                "symbol": sym, "ts": fdf.index[i], "ano": fdf.index[i].year,
                "direcao": direcao, "v1_score": score,
                "regime_down": regime_down, "entry": c[i],
            })
            for nome in CONFIGS:
                resultados[nome]["rot"].append(linha_rot[nome])
                resultados[nome]["pnl"].append(linha_pnl[nome])
                resultados[nome]["dir"].append(linha_dir[nome])
            n_sym += 1

        print(f"  {sym}: {n_sym} eventos", flush=True)
        del adf, a2, b2, fdf, X

    if not seqs:
        raise SystemExit("Nenhum evento gerado.")

    X_seq = np.stack(seqs).astype(np.float32)
    X_tab = np.stack(tabs).astype(np.float32)
    meta = pd.DataFrame(metas)

    # ── Rotulos derivados ───────────────────────────────────────────────────
    rotulos = {}
    for nome in CONFIGS:
        tb = np.array(resultados[nome]["rot"], dtype=np.int64)
        pnl = np.array(resultados[nome]["pnl"], dtype=np.float32)
        rotulos[f"tb_{nome}"] = tb                       # 0 stop | 1 tempo | 2 alvo
        rotulos[f"meta_{nome}"] = (tb == 2).astype(np.int64)   # bateu o alvo?
        rotulos[f"pnl_{nome}"] = pnl

        rotulos[f"dir_{nome}"] = np.array(resultados[nome]["dir"], dtype=np.int64)

    np.savez_compressed(os.path.join(DEST, "base.npz"),
                        X_seq=X_seq, X_tab=X_tab, **rotulos)
    meta.to_parquet(os.path.join(DEST, "meta.parquet"))
    with open(os.path.join(DEST, "colunas.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(nomes_tab or []))

    # ── Relatorio ───────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"  BASE V7 GERADA — {len(meta)} eventos")
    print(f"  X_seq {X_seq.shape} | X_tab {X_tab.shape}")
    print(f"  Periodo {meta['ts'].min():%Y-%m-%d} -> {meta['ts'].max():%Y-%m-%d}")
    print(f"  Direcao: {(meta['direcao']==1).sum()} compras / "
          f"{(meta['direcao']==-1).sum()} vendas")
    print(f"{'='*74}")
    print(f"\n  {'Configuracao':<16} {'%stop':>7} {'%tempo':>7} {'%alvo':>7} "
          f"{'expectancia':>12}")
    print("  " + "-" * 56)
    for nome in CONFIGS:
        tb = rotulos[f"tb_{nome}"]
        pnl = rotulos[f"pnl_{nome}"]
        print(f"  {nome:<16} {(tb==0).mean()*100:>6.1f}% {(tb==1).mean()*100:>6.1f}% "
              f"{(tb==2).mean()*100:>6.1f}% {pnl.mean()*100:>+11.4f}%")

    print(f"\n  Arquivos em {DEST}")
    print("  Proximo passo: python v7_modelos.py")


if __name__ == "__main__":
    main()
