import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "trading_state.db")

def init_db():
    """Inicializa a conexão com o banco de dados e cria a tabela de posições se não existir."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            qty REAL NOT NULL,
            avg_price REAL NOT NULL,
            highest_price REAL NOT NULL,
            sl_price REAL NOT NULL,
            tp_price REAL NOT NULL,
            last_buy_time TEXT,
            partial_tp_hit INTEGER NOT NULL,
            atr_distance REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def get_position(symbol: str) -> dict:
    """Retorna o estado da posição atual para um símbolo, ou dicionário vazio se não houver posição."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM positions WHERE symbol = ?', (symbol,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return {
        "symbol": symbol,
        "qty": 0.0,
        "avg_price": 0.0,
        "highest_price": 0.0,
        "sl_price": 0.0,
        "tp_price": 0.0,
        "last_buy_time": None,
        "partial_tp_hit": 0,
        "atr_distance": 0.0
    }

def update_position(position_data: dict):
    """Atualiza ou insere o estado de uma posição no banco de dados."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO positions (symbol, qty, avg_price, highest_price, sl_price, tp_price, last_buy_time, partial_tp_hit, atr_distance)
        VALUES (:symbol, :qty, :avg_price, :highest_price, :sl_price, :tp_price, :last_buy_time, :partial_tp_hit, :atr_distance)
        ON CONFLICT(symbol) DO UPDATE SET
            qty=excluded.qty,
            avg_price=excluded.avg_price,
            highest_price=excluded.highest_price,
            sl_price=excluded.sl_price,
            tp_price=excluded.tp_price,
            last_buy_time=excluded.last_buy_time,
            partial_tp_hit=excluded.partial_tp_hit,
            atr_distance=excluded.atr_distance
    ''', position_data)
    conn.commit()
    conn.close()

def clear_position(symbol: str):
    """Limpa o registro de posição de um símbolo (seta tudo para zero/null)."""
    empty_pos = {
        "symbol": symbol,
        "qty": 0.0,
        "avg_price": 0.0,
        "highest_price": 0.0,
        "sl_price": 0.0,
        "tp_price": 0.0,
        "last_buy_time": None,
        "partial_tp_hit": 0,
        "atr_distance": 0.0
    }
    update_position(empty_pos)

# Initialize the database file and tables upon importing
init_db()
