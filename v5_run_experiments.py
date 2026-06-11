"""
v5_run_experiments.py — Trader.AI V5.9: roda os 2 experimentos em sequencia
============================================================================

Experimento A "mais sinais"          : ALTA/QUEDA simetricos 0.4%
Experimento B "especialista em queda": ALTA 0.8% / QUEDA 0.4%

Pipeline (tudo automatico, ~6-10h no GTX 1650):
  1. v5_data_prep.py --dual          -> data_v5a/ (X+y_A) + data_v5b/ (y_B)
  2. v5_train.py (A)                 -> v5_model_a.pth
  3. v5_train.py (B)                 -> v5_model_b.pth
  4. v5_backtest.py A e B            -> v5_bt_{a,b}_{val,test}.txt
  5. Sentinela                       -> v5_experiments_done.txt

Acompanhe ao vivo:
  Get-Content v5_experiments.log -Wait
  Get-Content v5_training_a.log -Wait    (durante o treino A)
"""
import os
import sys
import subprocess
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable
LOG  = os.path.join(BASE, "v5_experiments.log")
DONE = os.path.join(BASE, "v5_experiments_done.txt")

# (nome, comando, arquivo_de_saida_ou_None, critico)
STEPS = [
    ("PREP DUAL (features + labels A/B)",
     [PY, "v5_data_prep.py", "--dual"], None, True),

    ("TREINO A — mais sinais (0.4% simetrico)",
     [PY, "v5_train.py", "--data", "data_v5a",
      "--model-out", "v5_model_a.pth",
      "--log", "v5_training_a.log", "--label", "V5.9-A"], None, False),

    ("TREINO B — especialista em queda (ALTA 0.8% / QUEDA 0.4%)",
     [PY, "v5_train.py", "--data", "data_v5a", "--y-dir", "data_v5b",
      "--model-out", "v5_model_b.pth",
      "--log", "v5_training_b.log", "--label", "V5.9-B"], None, False),

    ("BACKTEST A — validacao (Jul-Dez 2025)",
     [PY, "v5_backtest.py", "--model", "v5_model_a.pth", "--val"],
     "v5_bt_a_val.txt", False),

    ("BACKTEST A — teste (Jan-Mai 2026)",
     [PY, "v5_backtest.py", "--model", "v5_model_a.pth"],
     "v5_bt_a_test.txt", False),

    ("BACKTEST B — validacao (Jul-Dez 2025)",
     [PY, "v5_backtest.py", "--model", "v5_model_b.pth", "--val"],
     "v5_bt_b_val.txt", False),

    ("BACKTEST B — teste (Jan-Mai 2026)",
     [PY, "v5_backtest.py", "--model", "v5_model_b.pth"],
     "v5_bt_b_test.txt", False),
]


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    open(LOG, "w", encoding="utf-8").close()
    if os.path.exists(DONE):
        os.remove(DONE)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    log("=" * 60)
    log("V5.9 — EXPERIMENTOS A vs B")
    log("=" * 60)

    results = []
    skip_models = set()   # treinos que falharam -> pula backtests dependentes

    for name, cmd, outfile, critical in STEPS:
        # Pula backtest se o treino correspondente falhou
        model_dep = None
        for part in cmd:
            if part.endswith(".pth"):
                model_dep = part
        if model_dep and model_dep in skip_models:
            log(f"[PULADO] {name} (treino de {model_dep} falhou)")
            results.append((name, "PULADO"))
            continue

        log(f"[INICIO] {name}")
        t0 = datetime.now()

        stdout_target = None
        try:
            if outfile:
                stdout_target = open(os.path.join(BASE, outfile), "w",
                                     encoding="utf-8")
                proc = subprocess.run(cmd, cwd=BASE, env=env,
                                      stdout=stdout_target,
                                      stderr=subprocess.STDOUT)
            else:
                with open(LOG, "a", encoding="utf-8") as lf:
                    proc = subprocess.run(cmd, cwd=BASE, env=env,
                                          stdout=lf,
                                          stderr=subprocess.STDOUT)
            rc = proc.returncode
        except Exception as exc:
            log(f"[ERRO] {name}: {exc}")
            rc = -1
        finally:
            if stdout_target:
                stdout_target.close()

        dt_min = (datetime.now() - t0).total_seconds() / 60
        status = "OK" if rc == 0 else f"FALHOU(rc={rc})"
        log(f"[FIM] {name} — {status} em {dt_min:.0f}min")
        results.append((name, status))

        if rc != 0:
            if critical:
                log("[ABORTADO] Etapa critica falhou — encerrando pipeline.")
                break
            # marca o modelo desta etapa de treino como indisponivel
            for part in cmd:
                if part.endswith(".pth") and "--model-out" in cmd:
                    skip_models.add(part)

    with open(DONE, "w", encoding="utf-8") as f:
        f.write(f"Concluido em {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        for name, status in results:
            f.write(f"  [{status}] {name}\n")

    log("=" * 60)
    log("PIPELINE CONCLUIDO — veja v5_experiments_done.txt")
    log("=" * 60)


if __name__ == "__main__":
    main()
