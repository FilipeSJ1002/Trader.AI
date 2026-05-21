import itertools
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
import gc
import sys

def simulate(records, params):
    initial_balance = 10000.0
    virtual_usdt = initial_balance
    
    sl_mult = params['sl_mult']
    tp_mult = params['tp_mult']
    ml_thresh = params['ml_thresh']
    atr_thresh = params['atr_thresh']
    buy_thresh = params['buy_thresh']
    
    active = False
    qty = 0.0
    avg_price = 0.0
    highest_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    atr_value = 0.0
    bars_held = 0
    
    TAKER_FEE = 0.001
    
    for row in records:
        current_price = row['close']
        current_high = row['high']
        current_low = row['low']
        
        rsi = row['rsi']
        macd_hist = row['macd_hist']
        bb_lower = row['bb_lower']
        atr = row['atr']
        ema_200 = row['ema_200']
        ml_prob_success = row['ml_prob_success']
        atr_pct = atr / current_price if current_price > 0 else 0.0
        
        if active:
            bars_held += 1
            if current_price > highest_price:
                highest_price = current_price
            
            # Trailing Stop: Break-even agressivo (0.5 * ATR)
            if highest_price >= avg_price + (0.5 * atr_value):
                break_even_lock = avg_price * (1 + TAKER_FEE * 2)
                if break_even_lock > sl_price:
                    sl_price = break_even_lock
            
            sold = False
            
            if current_high >= tp_price:
                sell_price = tp_price
                sold = True
            
            if not sold and current_low <= sl_price:
                sell_price = sl_price
                sold = True
                
            if not sold:
                if 50 <= bars_held <= 70:
                    if current_price > avg_price * 1.004:
                        sell_price = current_price
                        sold = True
                elif bars_held > 70:
                    if current_price > avg_price:
                        sl_price = max(sl_price, avg_price * (1 + TAKER_FEE * 2))
                    else:
                        sell_price = current_price
                        sold = True
                        
            if sold:
                revenue = qty * sell_price * (1 - TAKER_FEE)
                virtual_usdt += revenue
                active = False
                qty = 0.0
                
        if not active:
            buy_score = 0
            if rsi < 30: buy_score += 40
            elif rsi < 40: buy_score += 20
            
            if current_price <= bb_lower: buy_score += 35
            
            prev_macd_hist = row['prev_macd_hist']
            if not np.isnan(prev_macd_hist) and macd_hist > prev_macd_hist: 
                buy_score += 25
            
            if row['Cup_and_Handle'] == 1: 
                buy_score += 40
            if (row['CDL_ENGULFING'] == 100 or row['CDL_HAMMER'] == 100 or row['CDL_MORNINGSTAR'] == 100) and current_price <= bb_lower * 1.01:
                buy_score += 30
                
            if current_price <= ema_200:
                buy_score = 0
            
            if buy_score >= buy_thresh and ml_prob_success >= ml_thresh and atr_pct > atr_thresh:
                invested_usdt = virtual_usdt * 0.30
                if invested_usdt > 11.0:
                    qty = (invested_usdt * (1 - TAKER_FEE)) / current_price
                    virtual_usdt -= invested_usdt
                    
                    active = True
                    avg_price = current_price
                    highest_price = current_price
                    
                    atr_val = atr if atr > 0 else (current_price * 0.002)
                    atr_value = atr_val
                    sl_price = current_price - (sl_mult * atr_val)
                    tp_price = current_price + (tp_mult * atr_val)
                    bars_held = 0

    final_equity = virtual_usdt + (qty * current_price) if active else virtual_usdt
    return ((final_equity - initial_balance) / initial_balance) * 100

def run():
    print("Loading data...")
    file_path = "data/BTCUSDT_1m.parquet"
    df = pd.read_parquet(file_path).tail(43200).copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
            
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
    df['Body_Size'] = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = df['high'] - np.maximum(df['open'], df['close'])
    df['Lower_Wick'] = np.minimum(df['open'], df['close']) - df['low']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 0.00001)
    
    for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
        df[col] = 0
        
    H1 = df['high'].shift(5).rolling(window=35).max()
    L1 = df['low'].shift(5).rolling(window=35).min()
    H2 = df['high'].rolling(window=5).max()
    df['Cup_and_Handle'] = ((H1 - L1) / H1 > 0.015).astype(int)
    
    df.dropna(inplace=True)
    cols = df.columns
    macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
    df['prev_macd_hist'] = df[macd_hist_col].shift(1)
    
    print("Loading ML...")
    model_path = "scalper_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        bbu_col = [c for c in cols if c.startswith('BBU_')][0]
        bbl_col = [c for c in cols if c.startswith('BBL_')][0]
        atr_col = [c for c in cols if c.startswith('ATRr_')][0]
        rsi_col = [c for c in cols if c.startswith('RSI_')][0]
        obv_col = [c for c in cols if c.startswith('OBV')][0]
        vwap_col = [c for c in cols if c.startswith('VWAP')][0]
        roc_5_col = [c for c in cols if c.startswith('ROC_5')][0]
        roc_15_col = [c for c in cols if c.startswith('ROC_15')][0]
        ema_200_col = [c for c in cols if c.startswith('EMA_200')][0]
        
        X = pd.DataFrame({
            'RSI': df[rsi_col],
            'MACDh': df[macd_hist_col],
            'ATR': df[atr_col],
            'Dist_BBU': (df['close'] - df[bbu_col]) / df['close'],
            'Dist_BBL': (df['close'] - df[bbl_col]) / df['close'],
            'ret_1': df['ret_1'],
            'ret_2': df['ret_2'],
            'ret_3': df['ret_3'],
            'OBV': df[obv_col],
            'VWAP': df[vwap_col],
            'ATR_pct': df[atr_col] / df['close'],
            'ROC_5': df[roc_5_col],
            'ROC_15': df[roc_15_col],
            'Body_Size': df['Body_Size'],
            'Upper_Wick': df['Upper_Wick'],
            'Lower_Wick': df['Lower_Wick'],
            'Wick_Ratio': df['Wick_Ratio'],
            'CDL_ENGULFING': df['CDL_ENGULFING'],
            'CDL_HAMMER': df['CDL_HAMMER'],
            'CDL_MORNINGSTAR': df['CDL_MORNINGSTAR'],
            'Cup_and_Handle': df['Cup_and_Handle']
        })
        probs = model.predict_proba(X)
        df['ml_prob_success'] = probs[:, 1]
    else:
        df['ml_prob_success'] = 1.0

    print("Preparing vectorized records...")
    df['rsi'] = df[rsi_col]
    df['macd_hist'] = df[macd_hist_col]
    df['bb_lower'] = df[bbl_col]
    df['atr'] = df[atr_col]
    df['ema_200'] = df[ema_200_col]
    
    records = df[['close', 'high', 'low', 'rsi', 'macd_hist', 'bb_lower', 'atr', 'ema_200', 'ml_prob_success', 'prev_macd_hist', 'Cup_and_Handle', 'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']].to_dict('records')

    print("Running greedy targeted optimization...")
    # Buscando TP curtos e Stops longos e vice-versa para encontrar a anomalia ganhadora
    grid = {
        'sl_mult': [0.5, 1.0, 1.5, 2.0, 3.0],
        'tp_mult': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        'ml_thresh': [0.1, 0.4, 0.5],
        'atr_thresh': [0.0005, 0.001],
        'buy_thresh': [45, 60]
    }
    
    keys = grid.keys()
    combinations = list(itertools.product(*grid.values()))
    
    best_return = -100
    best_params = None
    
    sys.stdout.flush()
    
    total = len(combinations)
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        ret = simulate(records, params)
        if i % 50 == 0:
            print(f"Progress: {i}/{total} (Current best: {best_return:.2f}%)")
            sys.stdout.flush()
            
        if ret > best_return:
            best_return = ret
            best_params = params
            print(f"New best: {best_return:.2f}% with {best_params}")
            sys.stdout.flush()
            if best_return > 3.0:
                print("Target >3% reached!")
                break
            
    print(f"Final best: {best_return:.2f}% -> {best_params}")
    sys.stdout.flush()

if __name__ == '__main__':
    run()
