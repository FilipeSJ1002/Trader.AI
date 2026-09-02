# -*- coding: utf-8 -*-
"""
oraculo/modelo.py — o classificador de regime e, sobretudo, como julga-lo
==========================================================================

A Sprint 1 mediu o alvo: **~56% de acurácia na direcao diaria** para o sistema
bater comprar-e-segurar. Este arquivo constroi o classificador e mede se ele
chega la — sem ligar em operacao nenhuma.

Tres armadilhas que este arquivo evita de proposito
---------------------------------------------------

1. ACURACIA CRUA ENGANA. Nos 2.132 dias mapeados, 1.120 foram de alta: um
   modelo que responde "BULL" sempre acerta 52,5% e nao sabe nada. Por isso a
   metrica principal aqui e a acuracia BALANCEADA (a media do acerto em dias de
   alta e em dias de baixa), e o DummyClassifier entra na tabela como piso.

2. VALIDACAO EMBARALHADA MENTE. Serie temporal nao pode ser embaralhada: dias
   vizinhos se parecem, e um teste que sorteia linhas ve o futuro pelo vizinho.
   Aqui a validacao e walk-forward com EMBARGO — um intervalo descartado entre
   treino e teste, porque as features usam janelas de ate 200 dias e sem o
   embargo o fim do treino e o comeco do teste compartilham as mesmas barras.

3. ESCOLHER O MELHOR DE MUITOS INFLA O RESULTADO. Testar cinco modelos e
   reportar o melhor e o mesmo erro do ranking da V8. Por isso a tabela mostra
   TODOS, com o desvio entre as dobras, e o limiar de decisao e o de 56% fixado
   ANTES — nao o melhor numero que aparecer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMBARGO_DIAS = 210      # maior que a maior janela de feature (SMA200)


def modelos() -> dict[str, object]:
    """
    Do mais simples ao mais flexivel.

    Com ~2.000 amostras e ~30 features, modelos flexiveis decoram. A regressao
    logistica entra como o candidato serio, nao como formalidade — e por isso
    que nao ha rede neural aqui: nao ha amostra para sustenta-la.
    """
    return {
        "sempre a classe maior": DummyClassifier(strategy="most_frequent"),
        "moeda estratificada": DummyClassifier(strategy="stratified",
                                               random_state=0),
        "regressao logistica": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
        ),
        "floresta aleatoria": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, max_depth=4,
                                   min_samples_leaf=40, class_weight="balanced",
                                   random_state=0, n_jobs=-1),
        ),
        "gradient boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                           learning_rate=0.05,
                                           min_samples_leaf=40,
                                           random_state=0),
        ),
    }


@dataclass(frozen=True, slots=True)
class Avaliacao:
    nome: str
    acuracia: float
    acuracia_erro: float
    balanceada: float
    balanceada_erro: float
    acerto_bull: float
    acerto_bear: float
    fracao_bull_previsto: float
    dobras: int
    n_teste: int

    def __str__(self) -> str:
        return (f"{self.nome:<24} {self.balanceada*100:>6.2f}% "
                f"± {self.balanceada_erro*100:>4.2f}")


def dobras_temporais(n: int, n_dobras: int = 5, embargo: int = EMBARGO_DIAS):
    """
    Walk-forward: treina no passado, testa no futuro imediato, com um vao.

    Cada dobra amplia o treino e desloca o teste para frente — que e como o
    modelo seria usado de verdade. O embargo entre os dois descarta os dias em
    que as janelas de feature do teste ainda tocam o periodo de treino.
    """
    tamanho_teste = n // (n_dobras + 1)
    for k in range(n_dobras):
        fim_treino = tamanho_teste * (k + 1)
        ini_teste = fim_treino + embargo
        fim_teste = min(ini_teste + tamanho_teste, n)
        if fim_teste - ini_teste < 60:
            continue
        yield np.arange(0, fim_treino), np.arange(ini_teste, fim_teste)


def avaliar(nome: str, modelo, X: np.ndarray, y: np.ndarray,
            n_dobras: int = 5) -> Avaliacao | None:
    """Roda o walk-forward e devolve as metricas com desvio entre dobras."""
    accs, bals, bulls, bears, fracs = [], [], [], [], []
    total_teste = 0

    for i_treino, i_teste in dobras_temporais(len(y), n_dobras):
        y_tr, y_te = y[i_treino], y[i_teste]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        modelo.fit(X[i_treino], y_tr)
        p = modelo.predict(X[i_teste])

        accs.append(float((p == y_te).mean()))
        acerto_bull = float((p[y_te == 1] == 1).mean())
        acerto_bear = float((p[y_te == 0] == 0).mean())
        bulls.append(acerto_bull)
        bears.append(acerto_bear)
        bals.append((acerto_bull + acerto_bear) / 2)
        fracs.append(float((p == 1).mean()))
        total_teste += len(y_te)

    if len(accs) < 2:
        return None

    def erro(v):
        return float(np.std(v, ddof=1) / np.sqrt(len(v)))

    return Avaliacao(
        nome=nome,
        acuracia=float(np.mean(accs)), acuracia_erro=erro(accs),
        balanceada=float(np.mean(bals)), balanceada_erro=erro(bals),
        acerto_bull=float(np.mean(bulls)), acerto_bear=float(np.mean(bears)),
        fracao_bull_previsto=float(np.mean(fracs)),
        dobras=len(accs), n_teste=total_teste,
    )


def preparar(tabela: pl.DataFrame, colunas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Converte a tabela em X e y, descartando dias sem rotulo."""
    limpa = tabela.filter(pl.col("y").is_not_null()).sort("dia")
    X = limpa.select(colunas).to_numpy().astype(np.float64)
    y = limpa["y"].to_numpy().astype(int)
    return X, y
