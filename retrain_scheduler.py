"""
retrain_scheduler.py — V4 (Fase 3.4)
Auto-retreino semanal dos modelos de entrada e saida.

Pode ser invocado:
  1. Diretamente:  python retrain_scheduler.py
  2. Via API:      POST /retrain  (main.py chama run_retrain())
  3. Agendado:     use Task Scheduler (Windows) ou cron (Linux) apontando para este script

O que faz:
  1. Baixa os dados mais recentes da Binance Vision (incrementalmente)
  2. Reprocessa os parquets atualizados
  3. Retreina scalper_model.pkl  (modelo de entrada — train_model.py)
  4. Retreina exit_model.pkl     (modelo de saida   — train_exit_model.py)
  5. Salva um log de retreino em retrain_log.json
"""
import os
import sys
import json
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RETRAIN_LOG_PATH = os.path.join(os.path.dirname(__file__), "retrain_log.json")


def _load_log() -> list:
    """Carrega o historico de retreinos."""
    if os.path.exists(RETRAIN_LOG_PATH):
        try:
            with open(RETRAIN_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(entries: list):
    """Salva o historico de retreinos."""
    try:
        with open(RETRAIN_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries[-50:], f, indent=2, ensure_ascii=False)  # mantém só os 50 mais recentes
    except Exception as e:
        logger.error(f"[RETRAIN] Erro ao salvar log: {e}")


def run_retrain(pairs: list = None, skip_download: bool = False) -> dict:
    """
    Executa o pipeline completo de retreino.

    Parâmetros
    ----------
    pairs        : lista de pares a retreinar. None = todos os 6 pares.
    skip_download: True para pular o download (útil em ambientes sem internet)

    Retorna
    -------
    dict com resultados do retreino (métricas WFV, timestamps, status)
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("TRADER.AI V4 — Pipeline de Retreino Automatico")
    logger.info(f"Inicio: {started_at}")
    logger.info("=" * 60)

    result = {
        "started_at":   started_at,
        "finished_at":  None,
        "status":       "running",
        "steps":        {},
        "entry_model":  {},
        "exit_model":   {},
        "error":        None,
    }

    # ── Passo 1: Download incremental de dados ─────────────────────────────
    if not skip_download:
        logger.info("[RETRAIN] Passo 1: Download de dados recentes...")
        try:
            from download_binance_data import run_etl
            run_etl(pairs=pairs, interval="1m")
            result["steps"]["download"] = "ok"
            logger.info("[RETRAIN] Download concluido.")
        except Exception as e:
            msg = f"Erro no download: {e}"
            logger.error(f"[RETRAIN] {msg}")
            result["steps"]["download"] = f"erro: {msg}"
            # Download falhou mas podemos continuar com dados existentes
    else:
        result["steps"]["download"] = "skipped"
        logger.info("[RETRAIN] Download pulado (skip_download=True).")

    # ── Passo 2: Re-processar parquets ────────────────────────────────────
    logger.info("[RETRAIN] Passo 2: Processando dados brutos -> parquet...")
    try:
        from processar_dados import processar_dados
        processar_dados()
        result["steps"]["processar_dados"] = "ok"
        logger.info("[RETRAIN] Processamento concluido.")
    except Exception as e:
        msg = f"Erro no processamento: {e}"
        logger.error(f"[RETRAIN] {msg}")
        result["steps"]["processar_dados"] = f"erro: {msg}"
        # Não fatal — se parquet já existe, usa o existente

    # ── Passo 3: Retreinar modelo de entrada (com loop de aprendizado) ────
    logger.info("[RETRAIN] Passo 3: Retreinando scalper_model (entrada)...")
    try:
        from train_model import train_model
        # Fase 5.6 — injeta a experiencia real do bot (trades ja fechados)
        extra = None
        try:
            from learn_from_trades import build_learning_dataset, stats
            extra = build_learning_dataset()
            result["learning_loop"] = stats()
        except Exception as le:
            logger.warning(f"[RETRAIN] Loop de aprendizado indisponivel: {le}")

        entry_meta = train_model(pairs=pairs, extra_samples=extra)
        result["entry_model"] = {
            "status":        "ok",
            "wfv_precision": entry_meta.get("wfv_precision"),
            "wfv_auc":       entry_meta.get("wfv_auc"),
            "pairs":         entry_meta.get("pairs_trained"),
            "real_trades_injected": (len(extra) if extra is not None else 0),
        }
        result["steps"]["train_entry"] = "ok"
        logger.info(f"[RETRAIN] Modelo de entrada retreinado | WFV AUC={entry_meta.get('wfv_auc')}")
    except Exception as e:
        msg = f"Erro no treino de entrada: {e}"
        logger.error(f"[RETRAIN] {msg}")
        result["entry_model"] = {"status": "erro", "message": msg}
        result["steps"]["train_entry"] = f"erro: {msg}"

    # ── Passo 4: Retreinar modelo de saida ────────────────────────────────
    logger.info("[RETRAIN] Passo 4: Retreinando exit_model (saida)...")
    try:
        from train_exit_model import train_exit_model
        train_exit_model(pairs=pairs)
        result["exit_model"] = {"status": "ok"}
        result["steps"]["train_exit"] = "ok"
        logger.info("[RETRAIN] Modelo de saida retreinado com sucesso.")
    except Exception as e:
        msg = f"Erro no treino de saida: {e}"
        logger.error(f"[RETRAIN] {msg}")
        result["exit_model"] = {"status": "erro", "message": msg}
        result["steps"]["train_exit"] = f"erro: {msg}"

    # ── Finaliza ───────────────────────────────────────────────────────────
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["status"] = "completed"

    # Persiste log
    log_entries = _load_log()
    log_entries.append(result)
    _save_log(log_entries)

    logger.info("[RETRAIN] Pipeline de retreino concluido!")
    logger.info(f"  Entrada: {result['entry_model']}")
    logger.info(f"  Saida:   {result['exit_model']}")
    logger.info("=" * 60)

    return result


def get_retrain_log(limit: int = 10) -> list:
    """Retorna os N registros mais recentes do log de retreino."""
    entries = _load_log()
    return entries[-limit:]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retreino automatico dos modelos Trader.AI V4")
    parser.add_argument("--pair", type=str, default=None,
                        help="Par especifico (ex: BTCUSDT). Padrao: todos.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Pula o download de dados (usa parquets existentes).")
    args = parser.parse_args()

    pairs = [args.pair.upper()] if args.pair else None
    run_retrain(pairs=pairs, skip_download=args.skip_download)
