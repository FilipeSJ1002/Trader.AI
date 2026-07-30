"""
learn_from_trades.py — V5 (Fase 5.6)
Loop de aprendizado contInuo.

Transforma os trades REAIS ja fechados pelo bot (tabela trade_features) em
amostras de treino rotuladas (features de entrada + resultado real), para
reinjetar a experiencia do proprio bot no retreino do modelo de ML.

Como funciona o ciclo completo:
  1. Bot abre posicao  -> execution.save_trade_features() grava as features de entrada
  2. Bot fecha posicao -> execution.close_trade_features() grava se deu lucro (outcome)
  3. Retreino semanal  -> build_learning_dataset() le esses trades e os injeta no
                          train() com peso maior, fazendo o modelo "aprender" com
                          os proprios acertos e erros.
"""
import numpy as np
import pandas as pd
from typing import cast

import database
from train_model import FEATURES

# Trades reais valem mais que candles historicos sinteticos.
# O modelo da mais importancia ao que REALMENTE aconteceu nas ordens do bot.
REAL_SAMPLE_WEIGHT = 5.0

# Minimo de trades reais fechados para valer a pena injetar no treino.
MIN_TRADES_TO_LEARN = 30


def build_learning_dataset(min_trades: int = MIN_TRADES_TO_LEARN,
                           min_pnl_abs: float = 0.05) -> pd.DataFrame | None:
    """
    Monta o DataFrame [FEATURES + target + sample_weight] a partir dos trades
    reais ja fechados. Retorna None se ainda nao houver trades suficientes.

    min_pnl_abs: ignora trades quase-breakeven (|pnl%| < limiar) que sao ruido.
    """
    samples = database.get_learning_samples(min_pnl_abs=min_pnl_abs)
    if len(samples) < min_trades:
        print(f"[LEARN] Apenas {len(samples)} trades reais fechados "
              f"(minimo {min_trades}). Loop de aprendizado aguardando mais dados.")
        return None

    df = pd.DataFrame(samples)
    # Garante todas as colunas de FEATURES (trades antigos podem nao ter MTF)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    df = cast(pd.DataFrame,
              df[FEATURES + ['target']].replace([np.inf, -np.inf], np.nan).dropna())
    if df.empty:
        return None

    df['sample_weight'] = REAL_SAMPLE_WEIGHT
    win_rate = df['target'].mean() * 100
    print(f"[LEARN] {len(df)} trades reais injetados no treino "
          f"(WIN={win_rate:.1f}% | peso={REAL_SAMPLE_WEIGHT}x).")
    return df


def stats() -> dict:
    """Resumo do que o bot ja aprendeu com os proprios trades."""
    samples = database.get_learning_samples()
    if not samples:
        return {"total_trades_aprendidos": 0, "win_rate": 0.0}
    df = pd.DataFrame(samples)
    return {
        "total_trades_aprendidos": len(df),
        "win_rate": round(df['target'].mean() * 100, 1),
        "vitorias": int(df['target'].sum()),
        "derrotas": int((1 - df['target']).sum()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(stats(), indent=2, ensure_ascii=False))
