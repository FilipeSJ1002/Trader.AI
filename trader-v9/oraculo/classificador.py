# -*- coding: utf-8 -*-
"""
oraculo/classificador.py — o oráculo de regime treinado, para uso ao vivo
==========================================================================

Treina o classificador sobre o histórico, grava em disco e responde ao vivo.
Satisfaz nucleo.protocolos.Oraculo, então entra no replay e na produção pela
mesma porta que os oráculos simulados.

O que este modelo é, medido em 31/08 e 03/09/2026
-------------------------------------------------
  acurácia balanceada  53,69% ± 0,90 no horizonte de 3 dias (t = 3,40)
  limiar para valer    58% na formulação de sobreposição
  confiança            NÃO carrega informação: nos 10% de dias mais confiantes
                       o acerto cai para 50,6%, abaixo da própria média

Ou seja: há sinal, ele é real, e é MENOR que o necessário. Este arquivo existe
para que o sistema possa ser operado e observado ao vivo, não porque a medição
recomende operá-lo. A recomendação registrada é a de não esperar retorno acima
de comprar-e-manter.

Por que o horizonte é 3 dias
----------------------------
Foi o único da varredura (1, 2, 3, 5, 7, 14 e 30 dias) cujo ganho sobre o
classificador ingênuo superou dois erros padrão. O de 14 dias tem média maior
(55,79%) e erro de 2,85, o que coloca seu limite inferior em 50,1%.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import sklearn
import polars as pl

from dados.visao import Historico
from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Regime
from oraculo.features import (colunas_de_feature, features_de_um_ativo,
                              montar_tabela)
from oraculo.modelo import modelos, preparar

HORIZONTE_DIAS = 3
CAMINHO_PADRAO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "modelos", "oraculo_3d.joblib",
)
NOME_MODELO = "gradient boosting"


def _rotulos(historicos: dict[str, Historico], dias, n: int) -> list:
    """Direção média do universo nos próximos `n` dias. O alvo do treino."""
    mapas = {}
    for sym, h in historicos.items():
        ts, _, _ = h.barras_entre(0, len(h) - 1)
        fech = h.em(len(h) - 1).serie("fechamento")
        d = (pl.DataFrame({"ts": ts, "f": fech})
             .group_by_dynamic("ts", every="1d", closed="left", label="left")
             .agg(pl.col("f").last()).drop_nulls())
        mapas[sym] = dict(zip(
            [x.astype("datetime64[us]").astype(datetime).date()
             for x in d["ts"].to_numpy()], d["f"].to_list()))

    saida = []
    for dia in dias:
        a, b = dia.date(), (dia + timedelta(days=n)).date()
        rets = [m[b] / m[a] - 1 for m in mapas.values()
                if m.get(a) and m.get(b)]
        saida.append(None if not rets else (1 if np.mean(rets) > 0 else 0))
    return saida


def treinar(
    historicos: dict[str, Historico],
    de: datetime = datetime(2019, 6, 1),
    ate: datetime | None = None,
    caminho: str = CAMINHO_PADRAO,
    referencia: str | None = None,
) -> dict:
    """
    Treina sobre todo o período e grava o modelo.

    Diferente da avaliação, aqui NÃO há conjunto de teste: o modelo destinado ao
    uso ao vivo é treinado com tudo o que existe. A avaliação honesta já foi
    feita em app/treinar.py, com walk-forward e embargo; retreinar com tudo é o
    procedimento correto para colocar em operação, e não invalida aquela medição.
    """
    ate = ate or datetime.utcnow()
    referencia = referencia or sorted(historicos)[0]

    tabela = montar_tabela(historicos, de, ate, referencia=referencia)
    colunas = colunas_de_feature(tabela)
    tabela = tabela.with_columns(pl.Series(
        "y", _rotulos(historicos, list(tabela["dia"]), HORIZONTE_DIAS),
        dtype=pl.Int8))
    X, y = preparar(tabela, colunas)

    modelo = modelos()[NOME_MODELO]
    modelo.fit(X, y)

    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    joblib.dump({
        "modelo": modelo,
        "colunas": colunas,
        "referencia": referencia,
        "horizonte": HORIZONTE_DIAS,
        "treinado_em": datetime.utcnow().isoformat(timespec="seconds"),
        "dias_de_treino": len(y),
        "fracao_alta": float(y.mean()),
        "acuracia_medida": 0.5369,      # de app/treinar.py, fora da amostra
        "erro_medido": 0.0090,
        "limiar_necessario": 0.58,      # de app/sobreposicao.py
        # Modelo em pickle NAO atravessa versoes do scikit-learn: em
        # 04/09/2026 um modelo treinado na 1.7.2 quebrou ao ser carregado na
        # 1.9.0 do servidor, com "ModuleNotFoundError: No module named
        # '_loss'". Gravar a versao permite avisar em vez de estourar.
        "sklearn": sklearn.__version__,
    }, caminho)
    return {"caminho": caminho, "dias": len(y), "colunas": len(colunas)}


class OraculoTreinado:
    """
    O classificador em uso. Satisfaz nucleo.protocolos.Oraculo.

    A decisão é por BLOCO de `horizonte` dias, não por dia: uma vez tomada, ela
    vale até o bloco terminar. Trocar de lado todo dia multiplicaria o custo de
    transação sem que o modelo tenha resolução para isso.
    """

    def __init__(self, historicos: dict[str, Historico],
                 caminho: str = CAMINHO_PADRAO):
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"{caminho} nao existe. Rode: python -m app.treinar_oraculo"
            )
        try:
            d = joblib.load(caminho)
        except Exception as e:
            raise SystemExit(
                "Nao consegui carregar o modelo.\n\n  arquivo : {caminho}\n  erro    : {tipo}: {erro}\n\n  Causa provavel: o modelo foi gravado com uma versao de scikit-learn\n  diferente da instalada aqui ({versao}). Modelo em pickle nao\n  atravessa versoes — em 04/09/2026 um treinado na 1.7.2 quebrou ao\n  ser carregado na 1.9.0, com ModuleNotFoundError: '_loss'.\n\n  Solucao: treine NESTA maquina, com os dados que ja estao em disco:\n      PYTHONPATH=. python -m app.treinar_oraculo".format(
                    caminho=caminho, tipo=type(e).__name__, erro=e,
                    versao=sklearn.__version__,
                )
            ) from e

        gravado = d.get("sklearn")
        if gravado and gravado != sklearn.__version__:
            print(f"[AVISO] modelo gravado com scikit-learn {gravado}, "
                  f"rodando na {sklearn.__version__}. Se os resultados "
                  f"parecerem estranhos, retreine: "
                  f"PYTHONPATH=. python -m app.treinar_oraculo")

        self.nome = f"treinado({d['horizonte']}d)"
        self._modelo = d["modelo"]
        self._colunas = d["colunas"]
        self._referencia = d["referencia"]
        self.horizonte = d["horizonte"]
        self.meta = d
        self._historicos = historicos
        self._cache: dict[object, Regime] = {}

    def _features(self, visao: VisaoDeMercado) -> np.ndarray:
        """Monta a linha de features do dia, na ordem exata do treino."""
        simbolos = sorted(self._historicos)
        dia = datetime(visao.ts.year, visao.ts.month, visao.ts.day)
        visoes = {s: self._historicos[s].em(self._historicos[s].indice_de(dia))
                  for s in simbolos}

        linha: dict[str, float] = {}
        linha.update({f"ref_{k}": v for k, v in
                      features_de_um_ativo(visoes[self._referencia]).items()})
        por_ativo = [features_de_um_ativo(visoes[s]) for s in simbolos]
        for chave in ("d_dist_sma20", "d_dist_sma50", "d_rsi", "d_ret1",
                      "d_ret7", "d_atr_rel"):
            vals = np.array([f[chave] for f in por_ativo], dtype=float)
            ok = vals[np.isfinite(vals)]
            linha[f"uni_{chave}_media"] = float(ok.mean()) if len(ok) else np.nan
            linha[f"uni_{chave}_disp"] = float(ok.std()) if len(ok) > 1 else np.nan
        ang = 2 * np.pi * dia.weekday() / 7
        linha["cal_sin"], linha["cal_cos"] = float(np.sin(ang)), float(np.cos(ang))

        return np.array([[linha.get(c, np.nan) for c in self._colunas]],
                        dtype=np.float64)

    def regime(self, visao: VisaoDeMercado) -> Regime:
        """BULL ou BEAR, uma decisão por bloco de `horizonte` dias."""
        bloco = visao.ts.toordinal() // self.horizonte
        if bloco in self._cache:
            return self._cache[bloco]
        X = self._features(visao)
        p = float(self._modelo.predict_proba(X)[0, 1])
        r = Regime.BULL if p > 0.5 else Regime.BEAR
        self._cache[bloco] = r
        return r

    def probabilidade(self, visao: VisaoDeMercado) -> float:
        """A probabilidade crua de alta. Para registro no log, não para decidir:
        a confiança do modelo foi medida e NÃO carrega informação."""
        return float(self._modelo.predict_proba(self._features(visao))[0, 1])
