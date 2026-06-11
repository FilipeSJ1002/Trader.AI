"""
v5_train.py — Trader.AI V5.1
Treino do classificador direcional (QUEDA/NEUTRO/ALTA).

Walk-forward:
  Treino:    2019-2023  | Validacao: 2024 | Teste: 2025+

Metricas-chave (o que realmente importa para operar):
  - Precisao da ALTA em alta confianca: quando o modelo diz "vai subir" com
    >=75%, quantas vezes acerta? (define se 20x faz sentido ou e suicidio)
  - Precisao da QUEDA: quando diz "vai cair", acerta? (sinal defensivo)

Acompanhe ao vivo:  Get-Content v5_training.log -Wait
"""
import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from datetime import datetime

from v5_model import get_model, get_device, count_params, N_CLASSES
from v5_data_prep import ASSETS, CLASS_NAMES

BATCH_SIZE       = 256        # voltado para 256 — 512 pode OOM no GTX 1650 com window=120
EPOCHS           = 60
LR               = 1e-4
LR_MIN           = 5e-6
PATIENCE         = 15
FOCAL_GAMMA      = 2.0        # focal loss: penaliza amostras faceis (NEUTRO facil)
CYCLES_PER_EPOCH = 3          # ciclos por epoch (mistura gradientes entre ativos)
MAX_PER_CYCLE    = 30000      # amostras por ativo por ciclo (3×6×30k = 540k/epoch)
MODEL_PATH  = "v5_model.pth"
DATA_DIR    = "data_v5"
Y_DIR       = None            # se definido, carrega y deste diretorio (X compartilhado)
LOG_PATH    = "v5_training.log"
RUN_LABEL   = "V5.8"


class FocalLoss(nn.Module):
    """
    Focal Loss para classificacao multi-classe desbalanceada.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Por que ajuda aqui:
      - CrossEntropy normal acumula gradiente principalmente dos exemplos NEUTRO
        (maioria), ensinando o modelo a prever NEUTRO quando em duvida.
      - O fator (1 - p_t)^gamma reduz a contribuicao dos exemplos 'faceis'
        (candles que o modelo ja classifica bem como NEUTRO) e forca o modelo
        a focar nos exemplos dificeis — justamente os sinais de ALTA e QUEDA.
      - O parametro 'weight' (alpha por classe) mantem a ponderacao inversa
        de frequencia do CrossEntropyLoss anterior.

    gamma=2 e o valor canonico da literatura (Lin et al., 2017 — RetinaNet).
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight   # alpha por classe (tensor [N_CLASSES])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE por amostra sem reducao (shape: [batch])
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        # Probabilidade da classe correta para cada amostra
        probs = torch.softmax(logits, dim=1)
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Fator focal: suprime exemplos faceis (p_t alto), amplifica dificeis
        focal_w = (1.0 - p_t) ** self.gamma
        return (focal_w * ce).mean()


def log(msg: str = ""):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_alt(split: str, sym: str):
    """Carrega sem copiar (from_numpy compartilha memoria com o array).
    Se Y_DIR estiver definido, X vem de DATA_DIR e y de Y_DIR
    (datasets compartilham as mesmas janelas, mudam apenas os rotulos)."""
    path = os.path.join(DATA_DIR, f"{split}_{sym}.npz")
    if not os.path.exists(path):
        return None, None
    with np.load(path) as data:
        X    = torch.from_numpy(np.ascontiguousarray(data["X"], dtype=np.float32))
        y_np = data["y"]
    if Y_DIR:
        ypath = os.path.join(Y_DIR, f"{split}_{sym}.npz")
        if not os.path.exists(ypath):
            return None, None
        with np.load(ypath) as yd:
            y_np = yd["y"]
        assert len(y_np) == len(X), \
            f"{sym}/{split}: X({len(X)}) != y({len(y_np)}) — datasets desalinhados"
    y = torch.from_numpy(y_np.astype(np.int64))
    return X, y


def load_y(split: str, sym: str):
    """Carrega APENAS os rotulos (sem o X de ~2GB) — para contar classes."""
    src = Y_DIR if Y_DIR else DATA_DIR
    path = os.path.join(src, f"{split}_{sym}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["y"]


def make_loader(X, y, shuffle=True, max_samples=None):
    """Usa SubsetRandomSampler — NAO copia X (evita estouro de memoria)."""
    ds = TensorDataset(X, y)
    if shuffle:
        n = len(y)
        if max_samples and n > max_samples:
            idx = torch.randperm(n)[:max_samples].tolist()
            return DataLoader(ds, batch_size=BATCH_SIZE,
                              sampler=SubsetRandomSampler(idx))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def class_weights() -> torch.Tensor:
    """Pesos inversos a frequencia de cada classe (NEUTRO domina)."""
    counts = np.zeros(N_CLASSES)
    for sym in ASSETS:
        y = load_y("train", sym)
        if y is not None:
            for k in range(N_CLASSES):
                counts[k] += int((y == k).sum())
            del y
    gc.collect()
    inv = counts.sum() / (counts + 1e-9)
    w = inv / inv.sum() * N_CLASSES
    log(f"Distribuicao treino: " +
        " | ".join(f"{CLASS_NAMES[k]}={int(counts[k])}" for k in range(N_CLASSES)))
    log(f"Pesos por classe: " +
        " | ".join(f"{CLASS_NAMES[k]}={w[k]:.2f}" for k in range(N_CLASSES)))
    return torch.tensor(w, dtype=torch.float32)


def directional_metrics(probs: np.ndarray, labels: np.ndarray):
    """
    Metricas focadas em operacao:
      - up_prec@conf: precisao da classe ALTA quando prob_ALTA >= conf
      - down_prec@conf: precisao da classe QUEDA quando prob_QUEDA >= conf
    """
    res = {}
    p_up   = probs[:, 2]
    p_down = probs[:, 0]
    for conf in [0.50, 0.60, 0.75]:
        # ALTA
        m = p_up >= conf
        if m.sum() > 0:
            up_prec = (labels[m] == 2).mean()
            res[f"up@{conf:.2f}"] = (up_prec, int(m.sum()))
        # QUEDA
        m = p_down >= conf
        if m.sum() > 0:
            dn_prec = (labels[m] == 0).mean()
            res[f"dn@{conf:.2f}"] = (dn_prec, int(m.sum()))
    acc = (probs.argmax(1) == labels).mean()
    return acc, res


def evaluate(model, split, device, criterion=None):
    """Avalia o modelo num split. Retorna (probs, labels, val_loss).
    val_loss e calculado apenas se criterion for passado."""
    model.eval()
    all_p, all_y = [], []
    tot_loss, nb = 0.0, 0
    with torch.no_grad():
        for sym in ASSETS:
            X, y = load_alt(split, sym)
            if X is None:
                continue
            for xb, yb in make_loader(X, y, shuffle=False):
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                if criterion is not None:
                    tot_loss += criterion(logits, yb).item()
                    nb += 1
                all_p.append(torch.softmax(logits, 1).cpu().numpy())
                all_y.append(yb.cpu().numpy())
            del X, y
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    val_loss = tot_loss / max(nb, 1) if criterion is not None else None
    return np.concatenate(all_p), np.concatenate(all_y), val_loss


def train():
    device = get_device()
    open(LOG_PATH, "w", encoding="utf-8").close()

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    log(f"\n=== Trader.AI {RUN_LABEL} — LR={LR} | train ate 2025-06 | val H2-2025 | test 2026 ===")
    log(f"Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"Device: {device} | Batch: {BATCH_SIZE} | Epochs ate {EPOCHS} | LR={LR} | FocalLoss(gamma={FOCAL_GAMMA})")
    log(f"Dados: X={DATA_DIR} | y={Y_DIR if Y_DIR else DATA_DIR} | Modelo out: {MODEL_PATH}")
    log(f"Ciclos/epoch: {CYCLES_PER_EPOCH} x {MAX_PER_CYCLE} amostras/ativo")

    # n_features (lê só o shape, sem manter em memória)
    n_features = None
    for sym in ASSETS:
        X0, _ = load_alt("train", sym)
        if X0 is not None:
            n_features = X0.shape[2]
            del X0; gc.collect()
            break
    if n_features is None:
        raise FileNotFoundError("Rode v5_data_prep.py primeiro.")

    weights = class_weights().to(device)
    model = get_model(n_features, device)
    log(f"Features: {n_features} | Janela: 120 | Modelo: {count_params(model):,} params\n")

    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)

    best_val_loss, patience_ct = float("inf"), 0
    log(f"{'Ep':>3} {'TrLoss':>7} {'VaLoss':>7} {'Acc':>6} {'ALTA@60':>14} {'QUEDA@60':>14}  Hora")
    log("-" * 74)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot_loss, nb = 0.0, 0
        # Ciclos: cada epoch faz CYCLES_PER_EPOCH passadas com ordem re-embaralhada.
        # Carrega 1 ativo por vez (pico ~1.5 GB RAM) — sem preload para evitar OOM.
        for _cycle in range(CYCLES_PER_EPOCH):
            order = ASSETS[:]; random.shuffle(order)
            for sym in order:
                X, y = load_alt("train", sym)
                if X is None:
                    continue
                for xb, yb in make_loader(X, y, shuffle=True, max_samples=MAX_PER_CYCLE):
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    tot_loss += loss.item(); nb += 1
                del X, y; gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        train_loss = tot_loss / max(nb, 1)

        probs, labels, val_loss = evaluate(model, "val", device, criterion)
        acc, res = directional_metrics(probs, labels)
        scheduler.step()

        up = res.get("up@0.60", (0.0, 0))
        dn = res.get("dn@0.60", (0.0, 0))
        up_s = f"{up[0]*100:.0f}%({up[1]})"
        dn_s = f"{dn[0]*100:.0f}%({dn[1]})"

        # Salva o modelo sempre que val_loss melhora (criterio padrao).
        mark = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_ct = 0
            mark = "  <- melhor"
        else:
            patience_ct += 1

        log(f"{epoch:>3} {train_loss:>7.4f} {val_loss:>7.4f} {acc*100:>5.1f}% "
            f"{up_s:>14} {dn_s:>14}  {datetime.now():%H:%M:%S}{mark}")

        if patience_ct >= PATIENCE:
            log(f"\nEarly stopping na epoch {epoch}. Val_loss nao melhorou em {PATIENCE} epochs.")
            break

    log(f"\n{'='*74}")
    log(f"Treino concluido. Melhor val_loss: {best_val_loss:.4f}")
    log(f"Modelo salvo em: {MODEL_PATH}")
    return best_val_loss


def evaluate_test():
    device = get_device()
    log("\n=== TESTE (2025+) — dados nunca vistos ===")
    n_features = None
    for sym in ASSETS:
        X0, _ = load_alt("test", sym)
        if X0 is not None:
            n_features = X0.shape[2]; del X0; gc.collect(); break
    model = load_model_safe(n_features, device)
    if model is None:
        log("Nenhum modelo salvo para avaliar.")
        return
    probs, labels, _ = evaluate(model, "test", device)
    acc, res = directional_metrics(probs, labels)
    log(f"\nAcuracia geral: {acc*100:.1f}%")
    log(f"\n{'Sinal':>12} {'Confianca':>10} {'Precisao':>10} {'Casos':>8}")
    log("-" * 44)
    for conf in [0.50, 0.60, 0.75]:
        u = res.get(f"up@{conf:.2f}")
        if u:
            log(f"{'ALTA':>12} {f'>={conf:.0%}':>10} {u[0]*100:>9.1f}% {u[1]:>8}")
    for conf in [0.50, 0.60, 0.75]:
        d = res.get(f"dn@{conf:.2f}")
        if d:
            log(f"{'QUEDA':>12} {f'>={conf:.0%}':>10} {d[0]*100:>9.1f}% {d[1]:>8}")


def load_model_safe(n_features, device):
    from v5_model import load_model
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH, n_features, device)


if __name__ == "__main__":
    import traceback
    import argparse

    ap = argparse.ArgumentParser(description="Trader.AI V5 — treino do classificador direcional")
    ap.add_argument("--data",      default=DATA_DIR,
                    help="Diretorio com os npz de X+y (padrao data_v5)")
    ap.add_argument("--y-dir",     default=None, dest="y_dir",
                    help="Diretorio alternativo apenas para os rotulos y "
                         "(X continua vindo de --data)")
    ap.add_argument("--model-out", default=MODEL_PATH, dest="model_out",
                    help="Arquivo de saida do modelo (padrao v5_model.pth)")
    ap.add_argument("--log",       default=LOG_PATH,
                    help="Arquivo de log (padrao v5_training.log)")
    ap.add_argument("--label",     default=RUN_LABEL,
                    help="Rotulo da run para o log (ex: V5.9-A)")
    args = ap.parse_args()

    # Rebind dos globais usados pelas funcoes de treino
    DATA_DIR   = args.data
    Y_DIR      = args.y_dir
    MODEL_PATH = args.model_out
    LOG_PATH   = args.log
    RUN_LABEL  = args.label

    t0 = datetime.now()
    try:
        train()
        evaluate_test()
    except Exception as exc:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n[CRASH] {exc}\n{traceback.format_exc()}\n")
        raise
    finally:
        log(f"\nTempo total: {(datetime.now()-t0).seconds}s")
