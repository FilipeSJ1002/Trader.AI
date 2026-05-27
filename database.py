"""
database.py — V4
Gerencia o banco de dados SQLite do Trader.AI.

Tabelas:
  positions   — estado atual das posições abertas por símbolo
  trades_log  — histórico completo de trades (compras e vendas)
                usado para calcular P&L, win rate e drawdown em tempo real
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_state.db")


# ─────────────────────────────────────────────────────────────────────────────
# Inicialização
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    """Cria (ou migra) as tabelas do banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tabela de posições abertas
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            symbol         TEXT PRIMARY KEY,
            qty            REAL    NOT NULL DEFAULT 0.0,
            avg_price      REAL    NOT NULL DEFAULT 0.0,
            highest_price  REAL    NOT NULL DEFAULT 0.0,
            sl_price       REAL    NOT NULL DEFAULT 0.0,
            tp_price       REAL    NOT NULL DEFAULT 0.0,
            last_buy_time  TEXT,
            partial_tp_hit INTEGER NOT NULL DEFAULT 0,
            atr_value      REAL    NOT NULL DEFAULT 0.0
        )
    ''')

    # Migração: renomear atr_distance → atr_value se a coluna antiga existir
    try:
        cursor.execute("ALTER TABLE positions RENAME COLUMN atr_distance TO atr_value")
        conn.commit()
    except Exception:
        pass  # coluna já tem o nome correto ou não existe

    # Tabela de log de trades — nova na V4
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol         TEXT    NOT NULL,
            side           TEXT    NOT NULL,   -- 'BUY' ou 'SELL'
            reason         TEXT,               -- ex: TAKE_PROFIT, STOP_LOSS, ML, TIME_STOP
            price          REAL    NOT NULL,
            qty            REAL    NOT NULL,
            usdt_value     REAL    NOT NULL,   -- qty * price
            fee_usdt       REAL    NOT NULL DEFAULT 0.0,
            pnl_pct        REAL,               -- lucro/prejuízo % em relação à compra (só em SELL)
            balance_after  REAL,               -- equity estimado após o trade
            timestamp      TEXT    NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Posições
# ─────────────────────────────────────────────────────────────────────────────
def get_position(symbol: str) -> dict:
    """Retorna o estado da posição atual para um símbolo."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM positions WHERE symbol = ?', (symbol,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {
        "symbol":        symbol,
        "qty":           0.0,
        "avg_price":     0.0,
        "highest_price": 0.0,
        "sl_price":      0.0,
        "tp_price":      0.0,
        "last_buy_time": None,
        "partial_tp_hit": 0,
        "atr_value":     0.0
    }


def update_position(position_data: dict):
    """Atualiza ou insere o estado de uma posição."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Garante que a chave usada seja atr_value (compatível com V4)
    data = dict(position_data)
    if 'atr_distance' in data:
        data['atr_value'] = data.pop('atr_distance')

    cursor.execute('''
        INSERT INTO positions
            (symbol, qty, avg_price, highest_price, sl_price, tp_price,
             last_buy_time, partial_tp_hit, atr_value)
        VALUES
            (:symbol, :qty, :avg_price, :highest_price, :sl_price, :tp_price,
             :last_buy_time, :partial_tp_hit, :atr_value)
        ON CONFLICT(symbol) DO UPDATE SET
            qty            = excluded.qty,
            avg_price      = excluded.avg_price,
            highest_price  = excluded.highest_price,
            sl_price       = excluded.sl_price,
            tp_price       = excluded.tp_price,
            last_buy_time  = excluded.last_buy_time,
            partial_tp_hit = excluded.partial_tp_hit,
            atr_value      = excluded.atr_value
    ''', data)
    conn.commit()
    conn.close()


def clear_position(symbol: str):
    """Zera a posição de um símbolo."""
    update_position({
        "symbol":        symbol,
        "qty":           0.0,
        "avg_price":     0.0,
        "highest_price": 0.0,
        "sl_price":      0.0,
        "tp_price":      0.0,
        "last_buy_time": None,
        "partial_tp_hit": 0,
        "atr_value":     0.0
    })


# ─────────────────────────────────────────────────────────────────────────────
# Log de Trades (V4)
# ─────────────────────────────────────────────────────────────────────────────
def log_trade(symbol: str, side: str, reason: str, price: float,
              qty: float, fee_usdt: float = 0.0,
              pnl_pct: float = None, balance_after: float = None):
    """
    Registra um trade no histórico.
    side: 'BUY' ou 'SELL'
    pnl_pct: percentual de lucro/prejuízo (apenas vendas)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO trades_log
            (symbol, side, reason, price, qty, usdt_value, fee_usdt,
             pnl_pct, balance_after, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        symbol, side, reason, price, qty,
        qty * price, fee_usdt, pnl_pct, balance_after,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_performance_summary() -> dict:
    """
    Retorna um resumo da performance do bot.
    Calcula: total de trades, win rate, P&L total, P&L médio, maior gain/loss,
    max drawdown e P&L dos últimos 7 dias.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM trades_log
        WHERE side = 'SELL' AND pnl_pct IS NOT NULL
        ORDER BY timestamp ASC
    ''')
    sells = [dict(r) for r in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) as cnt FROM trades_log WHERE side = 'BUY'")
    total_buys = cursor.fetchone()['cnt']

    conn.close()

    if not sells:
        return {
            "total_trades":   total_buys,
            "closed_trades":  0,
            "win_rate_pct":   0.0,
            "total_pnl_pct":  0.0,
            "avg_pnl_pct":    0.0,
            "best_trade_pct": 0.0,
            "worst_trade_pct": 0.0,
            "last_7d_pnl_pct": 0.0,
            "message":        "Nenhum trade fechado ainda."
        }

    pnls        = [s['pnl_pct'] for s in sells]
    wins        = [p for p in pnls if p > 0]
    win_rate    = len(wins) / len(pnls) * 100 if pnls else 0.0
    total_pnl   = sum(pnls)

    # P&L últimos 7 dias
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    last_7d = [s['pnl_pct'] for s in sells if s['timestamp'] >= cutoff]

    return {
        "total_trades":    total_buys,
        "closed_trades":   len(sells),
        "win_rate_pct":    round(win_rate, 2),
        "total_pnl_pct":   round(total_pnl, 4),
        "avg_pnl_pct":     round(total_pnl / len(pnls), 4) if pnls else 0.0,
        "best_trade_pct":  round(max(pnls), 4),
        "worst_trade_pct": round(min(pnls), 4),
        "last_7d_pnl_pct": round(sum(last_7d), 4),
        "last_7d_trades":  len(last_7d)
    }


def get_recent_trades(limit: int = 20) -> list:
    """Retorna os N trades mais recentes do log."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM trades_log
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# Inicializa o banco ao importar o módulo
init_db()
