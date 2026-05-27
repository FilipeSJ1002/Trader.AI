import os
import pandas as pd
import pandas_ta as ta
import joblib
import plotly.graph_objects as go
import numpy as np
import gc
from dotenv import load_dotenv
from train_model import FEATURES  # fonte única da lista de features — V4


def run_backtest():
    load_dotenv()
    file_path = os.getenv("DATA_FILE_PATH", "data/BTCUSDT_1m.parquet")

    print("Carregando dados históricos...")
    try:
        df = pd.read_parquet(file_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
    except Exception as e:
        print(f"❌ Erro ao ler dados: {e}")
        return

    # 30 dias de candles de 1M
    df = df.tail(43200).copy()

    print("Calculando Indicadores (Vetorizado)...")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=15, append=True)

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)

    df['vol_sma20'] = df['volume'].rolling(window=20).mean()

    # Geometria dos candles normalizada pelo close (V4)
    df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 0.00001)

    cdl = df.ta.cdl_pattern(name=["engulfing", "hammer", "morningstar"])
    if cdl is not None and isinstance(cdl, pd.DataFrame):
        for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
            if col in cdl.columns and col not in df.columns:
                df[col] = cdl[col]

    for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)

    H1 = df['high'].shift(5).rolling(window=35).max()
    L1 = df['low'].shift(5).rolling(window=35).min()
    H2 = df['high'].rolling(window=5).max()
    cup_depth       = (H1 - L1) / H1
    cup_edges_match = abs(H1 - H2) / H1
    handle_drop     = (H2 - df['close']) / H2
    cup_cond = (cup_depth > 0.015) & (cup_edges_match < 0.01) & (handle_drop > 0.002) & (handle_drop < 0.01)
    df['Cup_and_Handle'] = cup_cond.astype(int)

    df.dropna(inplace=True)

    cols         = df.columns
    rsi_col      = [c for c in cols if c.startswith('RSI_')][0]
    macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
    df['prev_macd_hist'] = df[macd_hist_col].shift(1)
    bbu_col      = [c for c in cols if c.startswith('BBU_')][0]
    bbl_col      = [c for c in cols if c.startswith('BBL_')][0]
    atr_col      = [c for c in cols if c.startswith('ATRr_')][0]
    ema_200_col  = [c for c in cols if c.startswith('EMA_200')][0]
    obv_col      = [c for c in cols if c.startswith('OBV')][0]
    vwap_col     = [c for c in cols if c.startswith('VWAP')][0]
    roc_5_col    = [c for c in cols if c.startswith('ROC_5')][0]
    roc_15_col   = [c for c in cols if c.startswith('ROC_15')][0]

    print("Processando Inferências de Machine Learning V4...")
    model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
    model = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)

    X = None
    if model:
        # Features normalizadas V4
        obv_pct   = df[obv_col].pct_change(1).fillna(0).clip(-1, 1)
        vwap_dist = (df['close'] - df[vwap_col]) / df[vwap_col]

        X = pd.DataFrame({
            'RSI':           df[rsi_col],
            'MACDh':         df[macd_hist_col],
            'ATR_pct':       df[atr_col] / df['close'],
            'Dist_BBU':      (df['close'] - df[bbu_col]) / df['close'],
            'Dist_BBL':      (df['close'] - df[bbl_col]) / df['close'],
            'ret_1':         df['ret_1'],
            'ret_2':         df['ret_2'],
            'ret_3':         df['ret_3'],
            'OBV_pct':       obv_pct,
            'VWAP_dist':     vwap_dist,
            'ROC_5':         df[roc_5_col],
            'ROC_15':        df[roc_15_col],
            'Body_Size':     df['Body_Size'],
            'Upper_Wick':    df['Upper_Wick'],
            'Lower_Wick':    df['Lower_Wick'],
            'Wick_Ratio':    df['Wick_Ratio'],
            'CDL_ENGULFING':    df['CDL_ENGULFING'],
            'CDL_HAMMER':       df['CDL_HAMMER'],
            'CDL_MORNINGSTAR':  df['CDL_MORNINGSTAR'],
            'Cup_and_Handle':   df['Cup_and_Handle']
        })[FEATURES]  # garante ordem exata do treino

        probs = model.predict_proba(X)
        df['ml_prob_fail']    = probs[:, 0]
        df['ml_prob_success'] = probs[:, 1]
    else:
        df['ml_prob_success'] = 1.0
        df['ml_prob_fail']    = 0.0

    if X is not None:
        del X
    gc.collect()

    print("Iniciando Simulação Event-Driven V4...")

    initial_balance = 10000.0
    virtual_usdt    = initial_balance

    position = {
        'active':        False,
        'qty':           0.0,
        'avg_price':     0.0,
        'highest_price': 0.0,
        'sl_price':      0.0,
        'tp_price':      0.0,
        'atr_value':     0.0,
        'buy_time':      None,
        'bars_held':     0
    }

    equity_curve  = []
    trades        = []
    buy_markers   = []
    sell_markers  = []

    TAKER_FEE = 0.001  # 0.1%

    for timestamp, row in df.iterrows():
        current_price = float(row['close'])
        current_high  = float(row['high'])
        current_low   = float(row['low'])

        rsi             = row[rsi_col]
        macd_hist       = row[macd_hist_col]
        bb_lower        = row[bbl_col]
        bb_upper        = row[bbu_col]
        atr             = row[atr_col]
        atr_pct         = atr / current_price if current_price > 0 else 0.0
        ema_200         = row[ema_200_col]
        ml_prob_success = row['ml_prob_success']
        ml_prob_fail    = row['ml_prob_fail']

        # ── Gerenciamento da posição aberta ───────────────────────────
        if position['active']:
            position['bars_held'] += 1

            if current_price > position['highest_price']:
                position['highest_price'] = current_price

            # Trailing Stop: Break-even após 1.5×ATR (V4)
            if position['highest_price'] >= position['avg_price'] + (1.5 * position['atr_value']):
                break_even_lock = position['avg_price'] * (1 + TAKER_FEE * 2)
                position['sl_price'] = max(position['sl_price'], break_even_lock)

            sold        = False
            sell_reason = ""
            sell_price  = 0.0
            sell_qty    = 0.0

            # Take Profit (3.0×ATR)
            if current_high >= position['tp_price']:
                sell_price  = position['tp_price']
                sell_qty    = position['qty']
                sold        = True
                sell_reason = "TAKE_PROFIT"

            # Stop Loss (1.2×ATR — V4)
            if not sold and current_low <= position['sl_price']:
                sell_price  = position['sl_price']
                sell_qty    = position['qty']
                sold        = True
                sell_reason = "STOP_LOSS/TRAILING_STOP"

            # Time-Stop: 4 horas (240 candles 1M) — alinhado com live (V4)
            if not sold:
                if 180 <= position['bars_held'] <= 240:
                    if current_price > position['avg_price'] * 1.004:
                        sell_price  = current_price
                        sell_qty    = position['qty']
                        sold        = True
                        sell_reason = "VENDA_TOTAL_TEMPORAL"
                elif position['bars_held'] > 240:
                    if current_price > position['avg_price']:
                        position['sl_price'] = max(position['sl_price'], position['avg_price'] * (1 + TAKER_FEE * 2))
                    else:
                        sell_price  = current_price
                        sell_qty    = position['qty']
                        sold        = True
                        sell_reason = "TEMPO_ESGOTADO"

            # Defesa Ativa do ML
            if not sold:
                if ml_prob_fail > 0.95:
                    sell_price  = current_price
                    sell_qty    = position['qty']
                    sold        = True
                    sell_reason = "VENDA_FORTE (ML PANIC)"
                elif ml_prob_fail > 0.92:
                    sell_qty  = position['qty'] * 0.50
                    sell_price = current_price
                    revenue   = sell_qty * sell_price * (1 - TAKER_FEE)
                    virtual_usdt += revenue
                    position['qty'] -= sell_qty
                    trades.append({'time': timestamp, 'type': 'VENDA_MODERADA (ML)', 'price': sell_price, 'revenue': revenue})
                    sell_markers.append((timestamp, sell_price))
                elif ml_prob_fail > 0.85:
                    sell_qty  = position['qty'] * 0.25
                    sell_price = current_price
                    revenue   = sell_qty * sell_price * (1 - TAKER_FEE)
                    virtual_usdt += revenue
                    position['qty'] -= sell_qty
                    trades.append({'time': timestamp, 'type': 'VENDA_LEVE (ML)', 'price': sell_price, 'revenue': revenue})
                    sell_markers.append((timestamp, sell_price))

            if sold:
                revenue = sell_qty * sell_price * (1 - TAKER_FEE)
                virtual_usdt += revenue
                trades.append({'time': timestamp, 'type': sell_reason, 'price': sell_price, 'revenue': revenue})
                sell_markers.append((timestamp, sell_price))

                position['active']    = False
                position['qty']       = 0.0
                position['avg_price'] = 0.0
                position['bars_held'] = 0

        # ── Busca de entrada ──────────────────────────────────────────
        if not position['active']:
            buy_score = 0
            if rsi < 30:   buy_score += 40
            elif rsi < 40: buy_score += 20

            if current_price <= bb_lower: buy_score += 35

            prev_macd_hist = row.get('prev_macd_hist', 0)
            if pd.notna(prev_macd_hist) and macd_hist > prev_macd_hist:
                buy_score += 25

            if row['Cup_and_Handle'] == 1:
                buy_score += 40
            if (row['CDL_ENGULFING'] == 100 or row['CDL_HAMMER'] == 100 or
                    row['CDL_MORNINGSTAR'] == 100) and current_price <= bb_lower * 1.01:
                buy_score += 30

            # Filtro de Tendência (EMA 200)
            if current_price <= ema_200:
                buy_score = 0

            decision = "NEUTRO"
            if   buy_score >= 85: decision = "COMPRA_FORTE"
            elif buy_score >= 60: decision = "COMPRA_MODERADA"

            # Filtros de entrada alinhados com o live (V4)
            if (decision in ["COMPRA_FORTE", "COMPRA_MODERADA"]
                    and ml_prob_success >= 0.60
                    and atr_pct > 0.003):
                invested_usdt = virtual_usdt * 0.90
                if invested_usdt > 11.0:
                    qty          = (invested_usdt * (1 - TAKER_FEE)) / current_price
                    virtual_usdt -= invested_usdt

                    position['active']        = True
                    position['qty']           = qty
                    position['avg_price']     = current_price
                    position['highest_price'] = current_price

                    # SL: 1.2×ATR | TP: 3.0×ATR (V4)
                    atr_val = atr if atr > 0 else (current_price * 0.002)
                    position['atr_value'] = atr_val
                    position['sl_price']  = current_price - (1.2 * atr_val)
                    position['tp_price']  = current_price + (3.0 * atr_val)

                    position['buy_time']  = timestamp
                    position['bars_held'] = 0

                    trades.append({'time': timestamp, 'type': 'BUY', 'price': current_price, 'cost': invested_usdt})
                    buy_markers.append((timestamp, current_price))

        current_equity = virtual_usdt + (position['qty'] * current_price) if position['active'] else virtual_usdt
        equity_curve.append({'time': timestamp, 'equity': current_equity})

    # ── Resultados ────────────────────────────────────────────────────
    eq_df        = pd.DataFrame(equity_curve).set_index('time')
    final_equity = eq_df['equity'].iloc[-1]
    total_return = ((final_equity - initial_balance) / initial_balance) * 100

    eq_df['cummax']   = eq_df['equity'].cummax()
    eq_df['drawdown'] = (eq_df['equity'] - eq_df['cummax']) / eq_df['cummax']
    max_dd            = eq_df['drawdown'].min() * 100

    buy_trades   = [t for t in trades if t['type'] == 'BUY']
    sell_trades  = [t for t in trades if t['type'] not in ('BUY',)]
    total_trades = len(buy_trades)

    # Taxa de acerto
    paired = []
    buy_iter = iter(buy_trades)
    sell_iter = iter([t for t in trades if 'revenue' in t and t['type'] != 'VENDA_MODERADA (ML)' and t['type'] != 'VENDA_LEVE (ML)'])
    for b in buy_trades:
        cost = b['cost']
        for s in sell_trades:
            if s['time'] >= b['time'] and 'revenue' in s:
                paired.append({'cost': cost, 'revenue': s['revenue']})
                break

    win_trades = sum(1 for p in paired if p['revenue'] > p['cost'])
    win_rate   = (win_trades / len(paired) * 100) if paired else 0.0

    print("\n" + "=" * 50)
    print("🏆  RESULTADOS DO BACKTEST V4")
    print("=" * 50)
    print(f"  Saldo Inicial:          ${initial_balance:>10.2f}")
    print(f"  Saldo Final:            ${final_equity:>10.2f}")
    print(f"  Retorno Total:          {total_return:>9.2f}%")
    print(f"  Total de Entradas:      {total_trades:>10}")
    print(f"  Win Rate estimado:      {win_rate:>9.1f}%")
    print(f"  Max Drawdown:           {max_dd:>9.2f}%")
    print("=" * 50)

    print("Gerando Relatório Visual (Plotly)...")
    plot_df = df.tail(10000)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['close'],
        mode='lines', name='Price', line=dict(color='gray', width=1)
    ))

    if buy_markers:
        valid_buys = [(x, y) for x, y in buy_markers if x >= plot_df.index[0]]
        if valid_buys:
            bx, by = zip(*valid_buys)
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode='markers', name='Buy',
                marker=dict(color='lime', size=12, symbol='triangle-up')
            ))

    if sell_markers:
        valid_sells = [(x, y) for x, y in sell_markers if x >= plot_df.index[0]]
        if valid_sells:
            sx, sy = zip(*valid_sells)
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode='markers', name='Sell/TP',
                marker=dict(color='red', size=12, symbol='triangle-down')
            ))

    fig.update_layout(
        title=f"Backtest V4 — Retorno: {total_return:.2f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%",
        xaxis_title="Time", yaxis_title="Price",
        template="plotly_dark", hovermode="x unified"
    )

    report_path = os.path.join(os.path.dirname(__file__), "backtest_report.html")
    fig.write_html(report_path)
    print(f"✅ Relatório salvo em: {report_path}")


if __name__ == "__main__":
    run_backtest()
