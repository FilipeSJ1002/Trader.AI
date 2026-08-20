# -*- coding: utf-8 -*-
"""
v7_modelos.py — o torneio: todos os modelos, todas as rotulagens
=================================================================

Treina uma bateria de modelos sobre a base do v7_dataset.py e ranqueia por
RETORNO FINANCEIRO — nao por acuracia.

Por que nao por acuracia: com tres classes onde 'stop' e ~59% dos casos, um
modelo que preveja sempre 'stop' acerta 59% e nunca opera. Acuracia alta e
facil e inutil. O que decide e a expectancia das operacoes que o modelo
manda fazer.

VALIDACAO TEMPORAL COM EMBARGO
------------------------------
Treino e teste sao separados por DATA, nunca aleatoriamente — em serie
temporal, embaralhar vaza o futuro para o treino. Alem disso, descartamos as
amostras cuja janela de resultado atravessa a fronteira (embargo), porque o
rotulo delas foi construido com precos que o teste tambem usa.

O QUE E MEDIDO
--------------
  acuracia, precisao, recall, F1   — as metricas classicas, para registro
  operacoes                        — quantas o modelo mandaria fazer
  taxa de acerto                   — quantas dessas bateram o alvo
  EXPECTANCIA                      — ganho medio por operacao, liquido de taxa
  retorno total                    — soma de tudo

Uso:
  python v7_modelos.py                       # tudo
  python v7_modelos.py --config B_3x1.5_2d   # so uma configuracao
  python v7_modelos.py --rapido              # pula os modelos lentos
"""
import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = "data_v7"
CORTE_TREINO = "2024-07-01"     # antes disso treina, depois testa
EMBARGO_DIAS = 3                # descarta o que atravessa a fronteira


def carrega():
    d = np.load(os.path.join(BASE, "base.npz"))
    meta = pd.read_parquet(os.path.join(BASE, "meta.parquet"))
    return d, meta


def split_temporal(meta):
    """Indices de treino e teste, com embargo na fronteira."""
    ts = pd.to_datetime(meta["ts"])
    corte = pd.Timestamp(CORTE_TREINO)
    fim_embargo = corte + pd.Timedelta(days=EMBARGO_DIAS)
    tr = np.where(ts < corte)[0]
    te = np.where(ts >= fim_embargo)[0]
    return tr, te


def modelos_tabulares(rapido=False):
    """Nome -> construtor. Ordem: dos baratos aos caros."""
    m = {
        "0-baseline (classe maior)": lambda: DummyClassifier(strategy="most_frequent"),
        "1-regressao logistica": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "2-random forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42,
            class_weight="balanced"),
        "3-extra trees": lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42,
            class_weight="balanced"),
    }
    if rapido:
        return m
    try:
        from lightgbm import LGBMClassifier
        m["4-lightgbm"] = lambda: LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, class_weight="balanced", verbose=-1,
            random_state=42)
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier
        m["5-xgboost"] = lambda: XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, verbosity=0)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        m["6-catboost"] = lambda: CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=5, verbose=0,
            random_seed=42, auto_class_weights="Balanced")
    except ImportError:
        pass
    return m


def transforma_minirocket(X_seq, tr, te):
    """
    MiniRocket: estado da arte em classificacao de series, treina em segundos.
    Espera (amostras, canais, tempo) — a nossa base e (amostras, tempo, canais).
    O transformador e ajustado SO no treino, para nao vazar o teste.
    """
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    Xt = np.transpose(X_seq, (0, 2, 1))
    mr = MiniRocketMultivariate(num_kernels=5000, random_state=42)
    mr.fit(Xt[tr])
    return np.asarray(mr.transform(Xt[tr])), np.asarray(mr.transform(Xt[te]))


def decide_operacoes(esquema, pred, proba, classes, direcao_te, limiar=0.5):
    """
    Converte a saida do modelo em 'opera / nao opera'.

      meta  -> opera se P(bateu o alvo) >= limiar
      tb    -> opera se a classe prevista for 'alvo' (2)
      dir   -> opera se a direcao prevista casar com a direcao da operacao
               (a operacao ja e LONG ou SHORT pelo regime; o modelo confirma)
    """
    if esquema == "meta":
        if proba is None:
            return pred == 1
        i1 = list(classes).index(1) if 1 in classes else None
        return proba[:, i1] >= limiar if i1 is not None else pred == 1
    if esquema == "tb":
        return pred == 2
    # dir: 2 = alta confirma compra, 0 = queda confirma venda
    return np.where(direcao_te == 1, pred == 2, pred == 0)


def avalia(nome, esquema, cfg, y_te, pred, proba, classes, pnl_te, direcao_te,
           segundos):
    opera = decide_operacoes(esquema, pred, proba, classes, direcao_te)
    n_ops = int(opera.sum())
    if n_ops == 0:
        exp = tot = acerto = 0.0
        se = float("nan")
    else:
        p = pnl_te[opera]
        exp, tot = float(p.mean()), float(p.sum())
        se = float(p.std(ddof=1) / np.sqrt(n_ops)) if n_ops > 1 else float("nan")
        acerto = float((p > 0).mean())
    return {
        "modelo": nome, "esquema": esquema, "config": cfg,
        "acuracia": float(accuracy_score(y_te, pred)),
        "precisao": float(precision_score(y_te, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_te, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "operacoes": n_ops, "taxa_acerto": acerto,
        "expectancia": exp, "erro_padrao": se,
        "t": exp / se if se and not np.isnan(se) and se > 0 else 0.0,
        "retorno_total": tot, "segundos": round(segundos, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="so uma configuracao")
    ap.add_argument("--esquema", default=None, choices=["tb", "meta", "dir"])
    ap.add_argument("--rapido", action="store_true", help="pula boosting e MiniRocket")
    ap.add_argument("--sem-minirocket", dest="sem_mr", action="store_true")
    a = ap.parse_args()

    d, meta = carrega()
    tr, te = split_temporal(meta)
    direcao = meta["direcao"].values
    ts = pd.to_datetime(meta["ts"])

    configs = ([a.config] if a.config else
               sorted({k.split("_", 1)[1] for k in d.files if k.startswith("pnl_")}))
    esquemas = [a.esquema] if a.esquema else ["meta", "tb", "dir"]

    print(f"\n{'='*92}")
    print(f"  TORNEIO DE MODELOS V7")
    print(f"  treino: {ts.iloc[tr].min():%Y-%m-%d} a {ts.iloc[tr].max():%Y-%m-%d} "
          f"({len(tr)} eventos)")
    print(f"  teste : {ts.iloc[te].min():%Y-%m-%d} a {ts.iloc[te].max():%Y-%m-%d} "
          f"({len(te)} eventos) | embargo de {EMBARGO_DIAS} dias")
    print(f"{'='*92}")

    X_tab = d["X_tab"]
    linhas = []
    cache_mr = None

    for cfg in configs:
        pnl_te = d[f"pnl_{cfg}"][te]
        # Referencia: operar TUDO que o V1 disparou, sem modelo nenhum
        linhas.append({
            "modelo": "* SEM MODELO (opera tudo)", "esquema": "-", "config": cfg,
            "acuracia": np.nan, "precisao": np.nan, "recall": np.nan, "f1": np.nan,
            "operacoes": len(pnl_te), "taxa_acerto": float((pnl_te > 0).mean()),
            "expectancia": float(pnl_te.mean()),
            "erro_padrao": float(pnl_te.std(ddof=1) / np.sqrt(len(pnl_te))),
            "t": float(pnl_te.mean() / (pnl_te.std(ddof=1) / np.sqrt(len(pnl_te)))),
            "retorno_total": float(pnl_te.sum()), "segundos": 0.0,
        })

        for esquema in esquemas:
            chave = f"{esquema}_{cfg}"
            if chave not in d.files:
                continue
            y = d[chave]
            y_tr, y_te = y[tr], y[te]
            if len(np.unique(y_tr)) < 2:
                continue

            for nome, cria in modelos_tabulares(a.rapido).items():
                t0 = time.time()
                try:
                    mdl = cria()
                    mdl.fit(X_tab[tr], y_tr)
                    pred = mdl.predict(X_tab[te])
                    proba = mdl.predict_proba(X_tab[te]) if hasattr(mdl, "predict_proba") else None
                    classes = getattr(mdl, "classes_", np.unique(y_tr))
                    linhas.append(avalia(nome, esquema, cfg, y_te, pred, proba,
                                         classes, pnl_te, direcao[te], time.time() - t0))
                except Exception as e:
                    print(f"    [erro] {nome} / {chave}: {str(e)[:90]}")

            if not (a.rapido or a.sem_mr):
                try:
                    t0 = time.time()
                    if cache_mr is None:
                        print("  (transformando com MiniRocket — uma vez so)", flush=True)
                        cache_mr = transforma_minirocket(d["X_seq"], tr, te)
                    Ztr, Zte = cache_mr
                    mdl = make_pipeline(StandardScaler(with_mean=False),
                                        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                                          class_weight="balanced"))
                    mdl.fit(Ztr, y_tr)
                    pred = mdl.predict(Zte)
                    linhas.append(avalia("7-minirocket+ridge", esquema, cfg, y_te,
                                         pred, None, np.unique(y_tr), pnl_te,
                                         direcao[te], time.time() - t0))
                except Exception as e:
                    print(f"    [erro] minirocket / {chave}: {str(e)[:90]}")

            print(f"  {cfg} / {esquema}: ok", flush=True)

    df = pd.DataFrame(linhas).sort_values("expectancia", ascending=False)
    df.to_csv(os.path.join(BASE, "resultados_modelos.csv"), index=False)

    print(f"\n{'='*100}")
    print("  RANQUEADO POR EXPECTANCIA (ganho medio por operacao, liquido de taxa)")
    print(f"{'='*100}")
    print(f"  {'modelo':<26}{'esq':<6}{'config':<15}{'ops':>6}{'acerto':>8}"
          f"{'expect.':>10}{'t':>7}{'total':>9}{'acur.':>7}")
    print("  " + "-" * 96)
    for _, r in df.head(30).iterrows():
        ac = "  -  " if pd.isna(r["acuracia"]) else f"{r['acuracia']*100:>5.1f}%"
        print(f"  {r['modelo']:<26}{r['esquema']:<6}{r['config']:<15}"
              f"{int(r['operacoes']):>6}{r['taxa_acerto']*100:>7.1f}%"
              f"{r['expectancia']*100:>+9.4f}%{r['t']:>+7.2f}"
              f"{r['retorno_total']*100:>+8.1f}%{ac:>7}")

    print(f"\n  Tabela completa: {BASE}/resultados_modelos.csv")
    print("  LEITURA: comparar cada modelo com a linha '* SEM MODELO' da mesma")
    print("           configuracao. Se nao superar, o modelo nao esta agregando.")


if __name__ == "__main__":
    main()
