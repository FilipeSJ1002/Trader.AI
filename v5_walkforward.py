"""
v5_walkforward.py — Trader.AI V5.9: Walk-Forward Retraining (Etapa 6)
======================================================================

Valida a hipotese: "modelo retreinado com dados recentes rende mais que
modelo congelado". Para cada trimestre, treina um modelo SO com dados
anteriores ao trimestre e opera o trimestre como dados nunca vistos.

  Fold 1: treina ate jul/2025  -> opera out-dez/2025  (Q4)
  Fold 2: treina ate out/2025  -> opera jan-mar/2026  (Q1)
  Fold 3: treina ate jan/2026  -> opera abr-mai/2026  (Q2)

Robustez (a GTX 1650 de 4GB divide VRAM com o desktop do usuario):
  - Treino usa --resume: crash de cuDNN/RAM retoma do ultimo checkpoint
  - Batch 192 (menos VRAM que o 256 padrao)
  - Cada etapa tem ate 3 tentativas com pausa de 90s (memoria assenta)
  - Dataset do fold so e apagado depois do treino dar certo

Acompanhe:  Get-Content v5_walkforward.log -Wait
            Get-Content v5_training_wf1.log -Wait   (fold em treino)
Resultados: v5_bt_wf{1,2,3}.txt vs v5_bt_frozen_q{1,2,3}.txt
"""
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime

# Console oculto do Windows usa cp1252 — sem isto, qualquer caractere fora
# da tabela derruba o processo inteiro no primeiro print()
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable
LOG  = os.path.join(BASE, "v5_walkforward.log")
DONE = os.path.join(BASE, "v5_walkforward_done.txt")

TRAIN_BATCH   = "192"   # 256 padrao estoura VRAM quando o desktop esta em uso
RETRIES       = 3
SLEEP_BETWEEN = 90      # segundos entre tentativas/etapas (memoria assenta)

FOLDS = [
    {
        "n": 1, "nome": "Q4-2025 (out-dez)",
        "train_end": "2025-07-31", "val_end": "2025-09-30",
        "test_from": "2025-10-01", "test_to": "2025-12-31",
    },
    {
        "n": 2, "nome": "Q1-2026 (jan-mar)",
        "train_end": "2025-10-31", "val_end": "2025-12-31",
        "test_from": "2026-01-01", "test_to": "2026-03-31",
    },
    {
        "n": 3, "nome": "Q2-2026 (abr-mai)",
        "train_end": "2026-01-31", "val_end": "2026-03-31",
        "test_from": "2026-04-01", "test_to": "2026-05-31",
    },
]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_once(cmd, outfile=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if outfile:
        with open(os.path.join(BASE, outfile), "w", encoding="utf-8") as out:
            return subprocess.run(cmd, cwd=BASE, env=env,
                                  stdout=out, stderr=subprocess.STDOUT).returncode
    with open(LOG, "a", encoding="utf-8") as lf:
        return subprocess.run(cmd, cwd=BASE, env=env,
                              stdout=lf, stderr=subprocess.STDOUT).returncode


def run_retry(label, cmd, outfile=None, tries=RETRIES):
    """Roda com ate `tries` tentativas. Pausa entre elas para a RAM assentar."""
    for attempt in range(1, tries + 1):
        rc = run_once(cmd, outfile)
        if rc == 0:
            return True
        log(f"[{label}] tentativa {attempt}/{tries} falhou (rc={rc})"
            + (f" — pausa de {SLEEP_BETWEEN}s e retry" if attempt < tries else ""))
        if attempt < tries:
            time.sleep(SLEEP_BETWEEN)
    return False


def main():
    open(LOG, "w", encoding="utf-8").close()
    if os.path.exists(DONE):
        os.remove(DONE)

    log("=" * 60)
    log("V5.9 — WALK-FORWARD RETRAINING (3 folds, receita B)")
    log(f"Batch {TRAIN_BATCH} | {RETRIES} tentativas/etapa | resume habilitado")
    log("=" * 60)

    results = []
    for f in FOLDS:
        n        = f["n"]
        data_dir = f"data_v5_wf{n}"        # padrao data_v5*/ ja esta no .gitignore
        model    = f"v5_model_wf{n}.pth"
        tlog     = f"v5_training_wf{n}.log"
        btout    = f"v5_bt_wf{n}.txt"

        log(f"----- FOLD {n}: {f['nome']} -----")

        # 1. Prep (pula automatico se os npz do fold ja existem)
        log(f"[FOLD {n}] prep -> {data_dir}/")
        ok = run_retry(f"FOLD {n} prep",
                       [PY, "v5_data_prep.py", "--fold-out", data_dir,
                        "--train-end", f["train_end"], "--val-end", f["val_end"]])
        if not ok:
            results.append((f["nome"], "PREP_FALHOU"))
            continue
        time.sleep(15)

        # 2. Treino (com resume: cada tentativa continua do checkpoint salvo)
        log(f"[FOLD {n}] treino -> {model} (acompanhe em {tlog})")
        t0 = datetime.now()
        ok = run_retry(f"FOLD {n} treino",
                       [PY, "v5_train.py", "--data", data_dir,
                        "--model-out", model, "--log", tlog,
                        "--label", f"V5.9-WF{n}",
                        "--resume", "--batch", TRAIN_BATCH])
        dt_h = (datetime.now() - t0).total_seconds() / 3600
        if not ok:
            log(f"[FOLD {n}] TREINO FALHOU apos {RETRIES} tentativas ({dt_h:.1f}h). "
                f"Dataset mantido em {data_dir}/ para retomada manual.")
            results.append((f["nome"], "TREINO_FALHOU"))
            continue
        log(f"[FOLD {n}] treino OK em {dt_h:.1f}h")
        time.sleep(15)

        # 3. Backtest do trimestre (dados nunca vistos pelo fold)
        log(f"[FOLD {n}] backtest {f['test_from']} -> {f['test_to']}")
        ok = run_retry(f"FOLD {n} backtest",
                       [PY, "v5_backtest.py", "--model", model,
                        "--from", f["test_from"], "--to", f["test_to"]],
                       outfile=btout)
        results.append((f["nome"], "OK" if ok else "BACKTEST_FALHOU"))
        log(f"[FOLD {n}] backtest {'OK' if ok else 'FALHOU'} -> {btout}")

        # 4. So apaga o dataset apos o fold completar (1.4GB cada — regeravel)
        shutil.rmtree(os.path.join(BASE, data_dir), ignore_errors=True)
        log(f"[FOLD {n}] dataset {data_dir}/ removido")
        time.sleep(15)

    # Baseline: modelo congelado (v5_model_b) nos mesmos trimestres
    log("----- BASELINE: modelo congelado v5_model_b.pth nos 3 trimestres -----")
    for f in FOLDS:
        btout = f"v5_bt_frozen_q{f['n']}.txt"
        ok = run_retry(f"BASELINE Q{f['n']}",
                       [PY, "v5_backtest.py", "--model", "v5_model_b.pth",
                        "--from", f["test_from"], "--to", f["test_to"]],
                       outfile=btout)
        log(f"[BASELINE Q{f['n']}] {'OK' if ok else 'FALHOU'} -> {btout}")
        time.sleep(15)

    with open(DONE, "w", encoding="utf-8") as fh:
        fh.write(f"Walk-forward concluido em {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        for nome, status in results:
            fh.write(f"  [{status}] {nome}\n")
        fh.write("\nCompare: v5_bt_wf{N}.txt (retreinado) vs v5_bt_frozen_q{N}.txt (congelado)\n")

    log("=" * 60)
    log("WALK-FORWARD CONCLUIDO — veja v5_walkforward_done.txt")
    log("=" * 60)


if __name__ == "__main__":
    main()
