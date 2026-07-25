"""
v5_model.py — Trader.AI V5.1
LSTM Bidirecional + Atencao para classificacao DIRECIONAL (3 classes).

Saida: logits para [QUEDA, NEUTRO, ALTA].
  - Probabilidade de ALTA  -> sinal de compra (alavancagem se alta confianca)
  - Probabilidade de QUEDA -> sinal defensivo (sair para stablecoin)

Input: (batch, WINDOW_SIZE, n_features)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 3   # 0=QUEDA, 1=NEUTRO, 2=ALTA


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out)
        weights = F.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights


class TradingLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 96,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, N_CLASSES)   # logits das 3 classes
        )

    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        context, attn = self.attention(lstm_out)
        context = self.dropout(context)
        logits = self.head(context)        # (batch, 3)
        return logits, attn


def get_model(n_features: int, device: str = "cpu") -> TradingLSTM:
    model = TradingLSTM(n_features=n_features, hidden_size=256,
                        n_layers=2, dropout=0.4)
    return model.to(device)


def get_device() -> str:
    # CUDA_VISIBLE_DEVICES="" faz is_available() retornar True em algumas versoes
    # do PyTorch, mas sem nenhum device valido -> testa de fato antes de usar.
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
            return "cuda"
    except Exception as e:
        print(f"[CPU] GPU indisponivel ({type(e).__name__}). Usando CPU.")
        return "cpu"
    print("[CPU] GPU nao detectada. Usando CPU.")
    return "cpu"


def load_model(path: str, n_features: int, device: str = "cpu") -> TradingLSTM:
    model = get_model(n_features, device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = get_device()
    n_feat = 18
    model = get_model(n_feat, device)
    print(f"\nTradingLSTM 3-classes | features={n_feat} | "
          f"params={count_params(model):,}")
    x = torch.randn(16, 60, n_feat).to(device)
    logits, attn = model(x)
    probs = torch.softmax(logits, dim=1)
    print(f"logits: {logits.shape} | probs soma=1: {probs.sum(1)[0].item():.3f}")
    print("[OK] Arquitetura 3-classes validada.")
