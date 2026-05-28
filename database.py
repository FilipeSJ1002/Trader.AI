"""
database.py — V4 (Fase 4)
Gerencia o banco de dados SQLite do Trader.AI.

Tabelas:
  positions   — estado atual das posições abertas por símbolo
  trades_log  — histórico completo de trades (compras e vendas)
  daily_stats — equity inicial do dia + flag de pausa por drawdown
"""
import sqlite3
import os
import json
from datetime import datetime, timezone

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
            atr_value      REAL    NOT NULL DEFAULT 0.0,
            tp1_price      REAL    NOT NULL DEFAULT 0.0,
            tp1_hit        INTEGER NOT NULL DEFAULT 0
        )
    ''')

    # Migrações — colunas novas adicionadas em versões anteriores
    migrations = [
        "ALTER TABLE positions RENAME COLUMN atr_distance TO atr_value",
        "ALTER TABLE positions ADD COLUMN tp1_price REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE positions ADD COLUMN tp1_hit   INTEGER NOT NULL DEFAULT 0",
    ]
    for sql in migrations:
        try:
            cursor.execute(sql)
            conn.commit()
        except Exception:
            pass  # coluna já existe ou renomeação desnecessária

    # Tabela de estatísticas diárias — Fase 4.2
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            date            TEXT PRIMARY KEY,  -- YYYY-MM-DD UTC
            start_equity    REAL DEFAULT 0.0,  -- equity ao abrir o dia
            min_equity      REAL DEFAULT 0.0,  -- mínimo intraday
            trading_paused  INTEGER DEFAULT 0  -- 1 = compras pausadas por drawdown
        )
    ''')

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

    # Tabela de features de entrada por trade — V5 (loop de aprendizado)
    # Guarda o vetor de features no momento da COMPRA e, ao fechar, o resultado.
    # Usada para reinjetar a experiencia real do bot no retreino do ML.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_features (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT    NOT NULL,
            entry_time    TEXT    NOT NULL,
            entry_price   REAL    NOT NULL,
            features_json TEXT    NOT NULL,   -- dict {feature: valor} no momento da entrada
            status        TEXT    NOT NULL DEFAULT 'open',  -- 'open' | 'closed'
            outcome       INTEGER,            -- 1 = trade lucrativo, 0 = perdedor (ao fechar)
            pnl_pct       REAL,               -- resultado % realizado
            exit_time     TEXT
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
        "symbol":         symbol,
        "qty":            0.0,
        "avg_price":      0.0,
        "highest_price":  0.0,
        "sl_price":       0.0,
        "tp_price":       0.0,
        "last_buy_time":  None,
        "partial_tp_hit": 0,
        "atr_value":      0.0,
        "tp1_price":      0.0,   # Fase 4.3 — TP em escada
        "tp1_hit":        0,     # Fase 4.3
    }


def update_position(position_data: dict):
    """Atualiza ou insere o estado de uma posição."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Garante que a chave usada seja atr_value (compatível com V4)
    data = dict(position_data)
    if 'atr_distance' in data:
        data['atr_value'] = data.pop('atr_distance')

    # Garante valores padrão para colunas novas (compatibilidade com dicts antigos)
    data.setdefault('tp1_price', 0.0)
    data.setdefault('tp1_hit',   0)

    cursor.execute('''
        INSERT INTO positions
            (symbol, qty, avg_price, highest_price, sl_price, tp_price,
             last_buy_time, partial_tp_hit, atr_value, tp1_price, tp1_hit)
        VALUES
            (:symbol, :qty, :avg_price, :highest_price, :sl_price, :tp_price,
             :last_buy_time, :partial_tp_hit, :atr_value, :tp1_price, :tp1_hit)
        ON CONFLICT(symbol) DO UPDATE SET
            qty            = excluded.qty,
            avg_price      = excluded.avg_price,
            highest_price  = excluded.highest_price,
            sl_price       = excluded.sl_price,
            tp_price       = excluded.tp_price,
            last_buy_time  = excluded.last_buy_time,
            partial_tp_hit = excluded.partial_tp_hit,
            atr_value      = excluded.atr_value,
            tp1_price      = excluded.tp1_price,
            tp1_hit        = excluded.tp1_hit
    ''', data)
    conn.commit()
    conn.close()


def clear_position(symbol: str):
    """Zera a posição de um símbolo."""
    update_position({
        "symbol":         symbol,
        "qty":            0.0,
        "avg_price":      0.0,
        "highest_price":  0.0,
        "sl_price":       0.0,
        "tp_price":       0.0,
        "last_buy_time":  None,
        "partial_tp_hit": 0,
        "atr_value":      0.0,
        "tp1_price":      0.0,
        "tp1_hit":        0,
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


# ─────────────────────────────────────────────────────────────────────────────
# Features de Trade (V5 — Loop de Aprendizado)
# ─────────────────────────────────────────────────────────────────────────────
def save_trade_features(symbol: str, entry_time: str, entry_price: float,
                        features: dict):
    """
    Salva o vetor de features no momento da COMPRA (status 'open').
    Chamado pelo execute_trade ao abrir uma posicao.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # limpa qualquer 'open' orfao do mesmo simbolo (so 1 posicao por simbolo)
    cursor.execute("DELETE FROM trade_features WHERE symbol = ? AND status = 'open'", (symbol,))
    cursor.execute('''
        INSERT INTO trade_features (symbol, entry_time, entry_price, features_json, status)
        VALUES (?, ?, ?, ?, 'open')
    ''', (symbol, entry_time, float(entry_price), json.dumps(features)))
    conn.commit()
    conn.close()


def close_trade_features(symbol: str, pnl_pct: float, won: bool = None,
                         exit_time: str = None):
    """
    Fecha o registro 'open' mais recente do simbolo, gravando o resultado.
    won: rotulo explicito (True/False). Se None, infere de pnl_pct > 0.
    Chamado quando a posicao e totalmente encerrada.
    """
    if exit_time is None:
        exit_time = datetime.now().isoformat()
    if won is None:
        outcome = 1 if (pnl_pct is not None and pnl_pct > 0) else 0
    else:
        outcome = 1 if won else 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE trade_features
           SET status = 'closed', outcome = ?, pnl_pct = ?, exit_time = ?
         WHERE id = (
            SELECT id FROM trade_features
             WHERE symbol = ? AND status = 'open'
             ORDER BY id DESC LIMIT 1
         )
    ''', (outcome, pnl_pct, exit_time, symbol))
    conn.commit()
    conn.close()


def get_learning_samples(min_pnl_abs: float = 0.0) -> list:
    """
    Retorna os trades reais ja fechados com suas features de entrada e o rotulo.
    Cada item: {**features, 'target': 0/1}. Usado pelo learn_from_trades.py.
    min_pnl_abs: ignora trades com |pnl_pct| abaixo do limiar (ruido ~breakeven).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT features_json, outcome, pnl_pct FROM trade_features
         WHERE status = 'closed' AND outcome IS NOT NULL
         ORDER BY id ASC
    ''')
    rows = cursor.fetchall()
    conn.close()

    samples = []
    for r in rows:
        if min_pnl_abs > 0 and r['pnl_pct'] is not None and abs(r['pnl_pct']) < min_pnl_abs:
            continue
        try:
            feats = json.loads(r['features_json'])
        except Exception:
            continue
        feats['target'] = int(r['outcome'])
        samples.append(feats)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Proteção de Drawdown Diário (Fase 4.2)
# ─────────────────────────────────────────────────────────────────────────────
DAILY_DRAWDOWN_LIMIT = 0.03   # 3% de perda máxima por dia → pausa compras


def _today_utc() -> str:
    """Retorna a data atual UTC no formato YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def get_daily_stats() -> dict:
    """Retorna (ou cria) o registro de hoje."""
    today = _today_utc()
    conn  = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (today,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return {
        'date':           today,
        'start_equity':   0.0,
        'min_equity':     0.0,
        'trading_paused': 0,
    }


def update_daily_equity(current_equity: float) -> bool:
    """
    Atualiza o equity do dia e verifica se o drawdown diário foi atingido.
    Retorna True se as compras devem ser pausadas.
    Se for o primeiro registro do dia, define start_equity.
    """
    today  = _today_utc()
    stats  = get_daily_stats()
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Primeiro registro do dia: define equity inicial
    if stats['start_equity'] == 0.0:
        stats['start_equity'] = current_equity
        stats['min_equity']   = current_equity

    stats['min_equity'] = min(stats['min_equity'], current_equity)

    # Verifica se o drawdown diário foi atingido
    if stats['start_equity'] > 0:
        drawdown = (stats['start_equity'] - current_equity) / stats['start_equity']
        if drawdown >= DAILY_DRAWDOWN_LIMIT:
            stats['trading_paused'] = 1

    cursor.execute('''
        INSERT INTO daily_stats (date, start_equity, min_equity, trading_paused)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            start_equity   = CASE WHEN excluded.start_equity > 0 AND start_equity = 0
                                  THEN excluded.start_equity ELSE start_equity END,
            min_equity     = excluded.min_equity,
            trading_paused = excluded.trading_paused
    ''', (today, stats['start_equity'], stats['min_equity'], stats['trading_paused']))
    conn.commit()
    conn.close()

    return bool(stats['trading_paused'])


def is_trading_paused() -> bool:
    """Retorna True se as compras estão pausadas por drawdown hoje."""
    stats = get_daily_stats()
    return bool(stats.get('trading_paused', 0))


def get_drawdown_summary() -> dict:
    """Retorna o resumo de drawdown do dia para o endpoint /performance."""
    stats = get_daily_stats()
    start = stats.get('start_equity', 0.0)
    low   = stats.get('min_equity', 0.0)
    if start > 0 and low > 0:
        dd = (start - low) / start * 100
    else:
        dd = 0.0
    return {
        'date':            stats.get('date', _today_utc()),
        'start_equity':    round(start, 2),
        'min_equity':      round(low, 2),
        'intraday_dd_pct': round(dd, 3),
        'trading_paused':  bool(stats.get('trading_paused', 0)),
        'dd_limit_pct':    DAILY_DRAWDOWN_LIMIT * 100,
    }


# Inicializa o banco ao importar o módulo
init_db()
