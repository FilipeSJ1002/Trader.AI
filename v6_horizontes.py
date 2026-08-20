# -*- coding: utf-8 -*-
"""
v6_horizontes.py — Trader.AI: a rede neural tem skill em ALGUM horizonte?
==========================================================================

O teste que decide se o modelo atual pode ser reaproveitado no V7.

Contexto (medido em 12/08/2026): a selecao de entradas acerta o alvo na mesma
taxa que um passeio aleatorio, em cinco proporcoes de risco/retorno. O edge
liquido e ~7 pontos-base contra 8 de custo. Antes de reconstruir qualquer
coisa, precisamos saber se existe skill em algum lugar.

O que este script mede — sem barreiras de TP/SL, sem taxa, sem alavancagem,
so o sinal cru:

    ret_dir(H) = retorno futuro em H minutos, NO SENTIDO da aposta
                 (+fwd se LONG, -fwd se SHORT)

Se o sistema tem poder preditivo, a media de ret_dir e positiva com
significancia. Se e zero dentro do erro, nao ha sinal — e nenhum ajuste de
execucao cria um.

Tres populacoes, para separar a contribuicao de cada portao:

    TODAS      todas as janelas (direcao dada so pelo regime SMA 24h)
    V1         janelas onde o gatilho tecnico disparou
    V1+NN      as que a rede aprovou (dir_conf >= 0.52) — o que opera de fato

Se V1+NN nao for melhor que V1, a rede nao esta agregando.
Se V1 nao for melhor que TODAS, o gatilho tecnico nao esta agregando.

Uso:
  python v6_horizontes.py                      # validacao (H2-2025)
  python v6_horizontes.py --from 2026-01-01 --to 2026-05-31   # teste
"""
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
from v5_backtest import v1_scores

HORIZONTES = [15, 30, 60, 120, 240, 360, 720, 1440]


def resumo(nome, ret_por_h, n):
    """Uma linha por populacao, com media, erro-padrao e t de cada horizonte."""
    print(f"\n  {nome}  (n = {n})")
    print(f"  {'horizonte':>10} {'media':>10} {'erro-padrao':>13} {'t':>7}  veredito")
    print("  " + "-" * 62)
    for h in HORIZONTES:
        r = ret_por_h[h]
        if len(r) < 30:
            print(f"  {h:>8}min {'(poucas)':>10}")
            continue
        m = r.mean()
        se = r.std(ddof=1) / np.sqrt(len(r))
        t = m / se if se > 0 else 0.0
        if abs(t) < 2:
            vered = "sem sinal (dentro do ruido)"
        elif t >= 2:
            vered = "SINAL POSITIVO"
        else:
            vered = "sinal NEGATIVO (aposta invertida)"
        print(f"  {h:>8}min {m*100:>+9.4f}% {'+/- ' + format(se*100, '.4f') + '%':>13} "
              f"{t:>+7.2f}  {vered}")


def main():
    ap = argparse.ArgumentParser(description="Skill do modelo por horizonte")
    ap.add_argument("--model", default="v5_model_b.pth")
    ap.add_argument("--from", dest="dt_from", default="2025-07-01")
    ap.add_argument("--to", dest="dt_to", default="2025-12-31")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--v1-thresh", dest="v1_thresh", type=int, default=60)
    ap.add_argument("--dirconf-min", dest="dirconf_min", type=float, default=0.52)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"
    btc = _load_parquet(BTC)
    model = None

    # ret[populacao][horizonte] -> lista de retornos direcionais
    pops = ("TODAS", "V1", "V1+NN")
    dados = {p: {h: [] for h in HORIZONTES} for p in pops}
    contagem = {p: 0 for p in pops}

    print(f"\nMedindo skill por horizonte ({a.dt_from} -> {a.dt_to})...")

    for sym in ASSETS:
        adf = _load_parquet(sym)
        com = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[com], btc.loc[com]
        fdf = _add_features(a2, b2).dropna()
        if len(fdf) < WINDOW_SIZE + max(HORIZONTES) + 10:
            continue

        X = fdf.values.astype(np.float32)
        cols = list(fdf.columns)
        close = a2["close"].reindex(fdf.index).values.astype(np.float64)
        sma24 = a2["close"].rolling(1440).mean().reindex(fdf.index).values

        if model is None:
            model = load_model(a.model, X.shape[1], device)

        janela = (fdf.index >= a.dt_from) & (fdf.index <= a.dt_to)
        idxs = np.where(janela)[0]
        idxs = idxs[(idxs >= WINDOW_SIZE) &
                    (idxs < len(close) - max(HORIZONTES) - 1)][::a.step]
        if len(idxs) == 0:
            continue

        # Classifica cada janela: TODAS / V1 / (NN decide depois)
        marcas = []
        for i in idxs:
            regime_down = bool(close[i] < sma24[i]) if not np.isnan(sma24[i]) else False
            direcao = "SHORT" if regime_down else "LONG"
            b_sc, s_sc, _ = v1_scores(X[i], X[i - 1], cols)
            tem_v1 = ((b_sc >= a.v1_thresh and not regime_down) or
                      (s_sc >= a.v1_thresh and regime_down))
            marcas.append((i, direcao, tem_v1))

        # Inferencia em lote
        for i0 in range(0, len(marcas), 512):
            lote = marcas[i0:i0 + 512]
            batch = np.stack([X[i - WINDOW_SIZE:i] for i, _, _ in lote])
            with torch.no_grad():
                logits, _ = model(torch.tensor(batch).to(device))
                p = torch.softmax(logits, 1).cpu().numpy()
            p_dn, p_up = p[:, 0], p[:, 2]
            p_dir = p_up + p_dn + 1e-9

            for k, (i, direcao, tem_v1) in enumerate(lote):
                dc = float((p_up[k] if direcao == "LONG" else p_dn[k]) / p_dir[k])
                sinal = 1.0 if direcao == "LONG" else -1.0
                rets = {h: sinal * (close[i + h] / close[i] - 1) for h in HORIZONTES}

                for h in HORIZONTES:
                    dados["TODAS"][h].append(rets[h])
                contagem["TODAS"] += 1
                if tem_v1:
                    for h in HORIZONTES:
                        dados["V1"][h].append(rets[h])
                    contagem["V1"] += 1
                    if dc >= a.dirconf_min:
                        for h in HORIZONTES:
                            dados["V1+NN"][h].append(rets[h])
                        contagem["V1+NN"] += 1

        print(f"  {sym} ok", flush=True)
        del adf, a2, b2, fdf, X

    print(f"\n{'='*74}")
    print(f"  SKILL POR HORIZONTE — {a.model}")
    print(f"  {a.dt_from} -> {a.dt_to} | retorno no sentido da aposta, SEM taxa")
    print(f"{'='*74}")

    for p in pops:
        if contagem[p] == 0:
            continue
        resumo(p, {h: np.array(dados[p][h]) for h in HORIZONTES}, contagem[p])

    print(f"\n{'='*74}")
    print("  COMO LER")
    print("    t = media / erro-padrao. |t| < 2 significa indistinguivel de zero.")
    print("    Para pagar 0,08% de taxa, a media precisa passar de +0,08%.")
    print("    Se V1+NN nao supera V1, a rede nao agrega no contexto de uso.")
    print(f"{'='*74}")


if __name__ == "__main__":
    main()
