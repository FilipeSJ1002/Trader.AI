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
        print(f"[ERRO] Falha ao ler dados: {e}")
        return

    # 30 dias de candles de 1M
    df = df.tail(43200).copy()

    print("Calculando Indicadores (Vetorizado)...")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.ema(length=20, append=True)   # necessario para EMA20_dist (V4)
    df.ta.ema(length=200, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=15, append=True)

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)

    # Geometria dos candles normalizada pelo close (V4)
    df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 0.00001)

    # Cup and Handle
    H1 = df['high'].shift(5).rolling(window=35).max()
    L1 = df['low'].shift(5).rolling(window=35).min()
    H2 = df['high'].rolling(window=5).max()
    cup_depth       = (H1 - L1) / H1
    cup_edges_match = abs(H1 - H2) / H1
    handle_drop     = (H2 - df['close']) / H2
    cup_cond = (cup_depth > 0.015) & (cup_edges_match < 0.01) & (handle_drop > 0.002) & (handle_drop < 0.01)
    df['Cup_and_Handle'] = cup_cond.astype(int)

    df.dropna(inplace=True)

    cols          = df.columns
    rsi_col       = [c for c in cols if c.startswith('RSI_')][0]
    macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
    df['prev_macd_hist'] = df[macd_hist_col].shift(1)
    bbu_col       = [c for c in cols if c.startswith('BBU_')][0]
    bbl_col       = [c for c in cols if c.startswith('BBL_')][0]
    atr_col       = [c for c in cols if c.startswith('ATRr_')][0]
    ema_20_col    = next((c for c in cols if c.startswith('EMA_20')), None)
    ema_200_col   = [c for c in cols if c.startswith('EMA_200')][0]
    obv_col       = [c for c in cols if c.startswith('OBV')][0]
    vwap_col      = [c for c in cols if c.startswith('VWAP')][0]
    roc_5_col     = [c for c in cols if c.startswith('ROC_5')][0]
    roc_15_col    = [c for c in cols if c.startswith('ROC_15')][0]

    # Novas features V4 (substituem CDL_ENGULFING/HAMMER/MORNINGSTAR — todos zeros sem TA-Lib)
    vol_sma20 = df['volume'].rolling(window=20).mean()
    df['vol_ratio']  = df['volume'] / (vol_sma20 + 1e-9)
    df['RSI_slope']  = df[rsi_col] - df[rsi_col].shift(3)
    df['EMA20_dist'] = (df['close'] - df[ema_20_col]) / (df['close'] + 1e-9) if ema_20_col else 0.0

    print("Processando Inferencias de Machine Learning V4...")
    model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
    model = None
    if os.path.exists(model_path):
        loaded = joblib.load(model_path)
        # V4: modelo salvo como dict {'model': ..., 'features': ...}
        model = loaded['model'] if isinstance(loaded, dict) else loaded

    X = None
    if model:
        obv_pct   = df[obv_col].pct_change(1).fillna(0).clip(-1, 1)
        vwap_dist = (df['close'] - df[vwap_col]) / (df[vwap_col] + 1e-9)

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
            'vol_ratio':     df['vol_ratio'],     # novo V4
            'RSI_slope':     df['RSI_slope'],     # novo V4
            'EMA20_dist':    df['EMA20_dist'],    # novo V4
            'Cup_and_Handle': df['Cup_and_Handle']
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

    print("Iniciando Simulacao Event-Driven V4 (Fase 4)...")

    initial_balance  = 10000.0
    virtual_usdt     = initial_balance
    total_equity_sim = initial_balance  # equity total (USDT + crypto)

    # Fase 4.2 — Proteção de drawdown diário no backtest
    day_start_equity: dict = {}   # {date_str: equity_inicial}
    daily_dd_paused:  dict = {}   # {date_str: bool}
    DAILY_DD_LIMIT = 0.03

    position = {
        'active':        False,
        'qty':           0.0,
        'avg_price':     0.0,
        'highest_price': 0.0,
        'sl_price':      0.0,
        'tp1_price':     0.0,    # Fase 4.3
        'tp1_hit':       False,  # Fase 4.3
        'tp_price':      0.0,
        'atr_value':     0.0,
        'buy_time':      None,
        'bars_held':     0
    }

    equity_curve = []
    trades       = []
    buy_markers  = []
    sell_markers = []

    TAKER_FEE = 0.001  # 0.1%

    # Fase 4.1 — Dimensionamento dinâmico (inline no backtest)
    BASE_ALLOC = 0.15
    MAX_ALLOC  = 0.25

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
        vol_ratio       = float(row.get('vol_ratio', 1.0))

        # Equity atual (para cálculos de drawdown e sizing)
        total_equity_sim = virtual_usdt + (position['qty'] * current_price if position['active'] else 0.0)

        # Fase 4.2 — Rastreia drawdown diário
        date_str = str(timestamp)[:10]
        if date_str not in day_start_equity:
            day_start_equity[date_str] = total_equity_sim
            daily_dd_paused[date_str]  = False
        else:
            start_eq = day_start_equity[date_str]
            if start_eq > 0 and (start_eq - total_equity_sim) / start_eq >= DAILY_DD_LIMIT:
                daily_dd_paused[date_str] = True

        # ── Gerenciamento da posição aberta ───────────────────────────
        if position['active']:
            position['bars_held'] += 1

            if current_price > position['highest_price']:
                position['highest_price'] = current_price

            # Trailing Stop: Break-even após 1.5×ATR
            if position['highest_price'] >= position['avg_price'] + (1.5 * position['atr_value']):
                break_even_lock = position['avg_price'] * (1 + TAKER_FEE * 2)
                position['sl_price'] = max(position['sl_price'], break_even_lock)

            sold        = False
            sell_reason = ""
            sell_price  = 0.0
            sell_qty    = 0.0

            # Defesa Ativa do ML (antes dos TPs)
            if not sold:
                if ml_prob_fail > 0.95:
                    sell_price  = current_price
                    sell_qty    = position['qty']
                    sold        = True
                    sell_reason = "ML_PANIC"
                elif ml_prob_fail > 0.92:
                    partial_qty  = position['qty'] * 0.50
                    revenue_p    = partial_qty * current_price * (1 - TAKER_FEE)
                    virtual_usdt += revenue_p
                    position['qty'] -= partial_qty
                    trades.append({'time': timestamp, 'type': 'ML_MODERADA', 'price': current_price, 'revenue': revenue_p})
                    sell_markers.append((timestamp, current_price))

            # Fase 4.3 — TP1: 1.5×ATR — vende 40%, sobe SL para breakeven
            if not sold and not position['tp1_hit'] and current_high >= position['tp1_price']:
                tp1_qty      = position['qty'] * 0.40
                revenue_tp1  = tp1_qty * position['tp1_price'] * (1 - TAKER_FEE)
                virtual_usdt += revenue_tp1
                position['qty']      -= tp1_qty
                position['tp1_hit']   = True
                be_price = position['avg_price'] * 1.002
                position['sl_price'] = max(position['sl_price'], be_price)
                trades.append({'time': timestamp, 'type': 'TP1_ESCADA', 'price': position['tp1_price'], 'revenue': revenue_tp1})
                sell_markers.append((timestamp, position['tp1_price']))

            # TP2 final (3.0×ATR)
            if not sold and current_high >= position['tp_price']:
                sell_price  = position['tp_price']
                sell_qty    = position['qty']
                sold        = True
                sell_reason = "TAKE_PROFIT_TP2"

            # Stop Loss (1.2×ATR)
            if not sold and current_low <= position['sl_price']:
                sell_price  = position['sl_price']
                sell_qty    = position['qty']
                sold        = True
                sell_reason = "STOP_LOSS"

            # Time-Stop: 4 horas (240 candles)
            if not sold:
                if 180 <= position['bars_held'] <= 240:
                    if current_price > position['avg_price'] * 1.004:
                        sell_price  = current_price
                        sell_qty    = position['qty']
                        sold        = True
                        sell_reason = "VENDA_TEMPORAL"
                elif position['bars_held'] > 240:
                    sell_price  = current_price
                    sell_qty    = position['qty']
                    sold        = True
                    sell_reason = "TEMPO_ESGOTADO"

            if sold and sell_qty > 0:
                revenue = sell_qty * sell_price * (1 - TAKER_FEE)
                virtual_usdt += revenue
                trades.append({'time': timestamp, 'type': sell_reason, 'price': sell_price, 'revenue': revenue})
                sell_markers.append((timestamp, sell_price))

                position['active']    = False
                position['qty']       = 0.0
                position['avg_price'] = 0.0
                position['bars_held'] = 0
                position['tp1_hit']   = False

        # ── Busca de entrada ──────────────────────────────────────────
        if not position['active']:
            # Fase 4.2 — Não compra se drawdown diário atingido
            if daily_dd_paused.get(date_str, False):
                pass
            else:
                buy_score = 0
                if rsi < 30:   buy_score += 40
                elif rsi < 40: buy_score += 20
                if current_price <= bb_lower: buy_score += 35

                prev_macd_hist = row.get('prev_macd_hist', 0)
                if pd.notna(prev_macd_hist) and macd_hist > prev_macd_hist:
                    buy_score += 25
                if row['Cup_and_Handle'] == 1:
                    buy_score += 40
                if vol_ratio >= 1.5:
                    buy_score += 15

                # Filtro de tendência (EMA 200)
                if current_price <= ema_200:
                    buy_score = 0

                decision = "NEUTRO"
                if   buy_score >= 85: decision = "COMPRA_FORTE"
                elif buy_score >= 60: decision = "COMPRA_MODERADA"

                if (decision in ["COMPRA_FORTE", "COMPRA_MODERADA"]
                        and ml_prob_success >= 0.60
                        and atr_pct > 0.003):

                    # Fase 4.1 — Dimensionamento dinâmico
                    ml_mult  = 1.0 + (ml_prob_success - 0.60) / 0.40
                    dec_mult = 1.0 if decision == "COMPRA_FORTE" else 0.75
                    invested = total_equity_sim * BASE_ALLOC * ml_mult * dec_mult
                    invested = min(invested, total_equity_sim * MAX_ALLOC)
                    invested = min(invested, virtual_usdt * 0.90)

                    if invested > 11.0:
                        qty          = (invested * (1 - TAKER_FEE)) / current_price
                        virtual_usdt -= invested

                        atr_val = atr if atr > 0 else (current_price * 0.002)
                        position.update({
                            'active':        True,
                            'qty':           qty,
                            'avg_price':     current_price,
                            'highest_price': current_price,
                            'atr_value':     atr_val,
                            'sl_price':      current_price - (1.2 * atr_val),
                            'tp1_price':     current_price + (1.5 * atr_val),   # Fase 4.3
                            'tp1_hit':       False,
                            'tp_price':      current_price + (3.0 * atr_val),
                            'buy_time':      timestamp,
                            'bars_held':     0,
                        })

                        trades.append({'time': timestamp, 'type': 'BUY', 'price': current_price, 'cost': invested})
                        buy_markers.append((timestamp, current_price))

        current_equity = virtual_usdt + (position['qty'] * current_price if position['active'] else 0.0)
        equity_curve.append({'time': timestamp, 'equity': current_equity})

    # ── Resultados ────────────────────────────────────────────────────
    eq_df        = pd.DataFrame(equity_curve).set_index('time')
    final_equity = eq_df['equity'].iloc[-1]
    total_return = ((final_equity - initial_balance) / initial_balance) * 100

    eq_df['cummax']   = eq_df['equity'].cummax()
    eq_df['drawdown'] = (eq_df['equity'] - eq_df['cummax']) / eq_df['cummax']
    max_dd            = eq_df['drawdown'].min() * 100

    buy_trades   = [t for t in trades if t['type'] == 'BUY']
    total_trades = len(buy_trades)

    # ── Taxa de acerto corrigida — acumula TP1 parcial + saida final por trade ──
    # Agrupa todos os eventos de trade em ciclos: 1 BUY → N sells (TP1 + final)
    FINAL_EXIT_TYPES = {
        'TAKE_PROFIT_TP2', 'STOP_LOSS', 'ML_PANIC',
        'VENDA_TEMPORAL', 'TEMPO_ESGOTADO'
    }
    trade_cycles = []
    current_cycle = None
    for t in trades:
        if t['type'] == 'BUY':
            current_cycle = {'cost': t['cost'], 'revenue': 0.0, 'closed': False}
            trade_cycles.append(current_cycle)
        elif current_cycle is not None and 'revenue' in t:
            current_cycle['revenue'] += t['revenue']
            if t['type'] in FINAL_EXIT_TYPES:
                current_cycle['closed'] = True
                current_cycle = None  # próximo BUY abre novo ciclo

    closed_cycles = [c for c in trade_cycles if c['closed']]
    win_trades    = sum(1 for c in closed_cycles if c['revenue'] > c['cost'])
    win_rate      = (win_trades / len(closed_cycles) * 100) if closed_cycles else 0.0

    # Contagem por tipo de saida
    exit_counts: dict = {}
    for t in trades:
        if t['type'] != 'BUY':
            exit_counts[t['type']] = exit_counts.get(t['type'], 0) + 1

    print("\n" + "=" * 55)
    print("RESULTADOS DO BACKTEST V4")
    print("=" * 55)
    print(f"  Saldo Inicial:          ${initial_balance:>10.2f}")
    print(f"  Saldo Final:            ${final_equity:>10.2f}")
    print(f"  Retorno Total:          {total_return:>9.2f}%")
    print(f"  Max Drawdown:           {max_dd:>9.2f}%")
    print(f"  Total de Entradas:      {total_trades:>10}")
    print(f"  Trades Fechados:        {len(closed_cycles):>10}")
    print(f"  Win Rate:               {win_rate:>9.1f}%")
    print("-" * 55)
    print("  Saidas por tipo:")
    for etype, cnt in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:<25} {cnt:>5}")
    print("-" * 55)
    # P&L por ciclo fechado
    if closed_cycles:
        pnls = [(c['revenue'] - c['cost']) / c['cost'] * 100 for c in closed_cycles]
        print(f"  P&L medio por trade:    {sum(pnls)/len(pnls):>8.2f}%")
        print(f"  Melhor trade:           {max(pnls):>8.2f}%")
        print(f"  Pior trade:             {min(pnls):>8.2f}%")
    print("=" * 55)

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
    print(f"[OK] Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    run_backtest()
