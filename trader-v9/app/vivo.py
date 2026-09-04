# -*- coding: utf-8 -*-
"""
app/vivo.py — a sobreposição em operação
=========================================

Uma decisão a cada três dias: comprado nos seis ativos, ou em caixa.

O que este programa NÃO faz
---------------------------
Não alavanca, não vende a descoberto, não usa stop nem alvo, não escolhe ativo
e não escolhe hora. Cada uma dessas ausências corresponde a um mecanismo que
foi medido e não se sustentou, ou que causou incidente:

  alavancagem      de 1x para 2x o retorno cai e o rebaixamento dobra (02/09)
  venda a descoberto  o motor Bear nunca superou o custo em medição nenhuma
  stop na corretora   foi o que falhou em 19/08, deixando posições descobertas
  escolha de ativo    peso igual venceu toda tentativa de seleção

O que esperar, honestamente
---------------------------
A medição deste arranjo, com a acurácia real de 53,69%, deu +146% em 3,5 anos
(~2,2% ao mês) com rebaixamento de 52,8%. O erro é de ±60 pontos percentuais e
o resultado NÃO se distingue estatisticamente de um controle aleatório. Comprar
e manter os mesmos ativos rendeu +215,5% no mesmo período.

Este programa existe para observar o sistema ao vivo, não porque a medição
recomende operá-lo esperando lucro.

Uso:
  python -m app.vivo                    # dry-run: decide e mostra, nao envia
  python -m app.vivo --armar            # envia ordens de verdade (testnet)
  python -m app.vivo --armar --real     # PRODUCAO — exige confirmacao digitada
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from dados.atualizar import atualizar, defasagem
from dados.fonte import FonteParquet, ler_config
from execucao.carteira import CarteiraBinance
from nucleo.tipos import Regime
from oraculo.classificador import CAMINHO_PADRAO, OraculoTreinado

ARQUIVO_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "v9_vivo.log",
)


def registrar(msg: str = "") -> None:
    linha = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] {msg}" if msg else ""
    print(linha, flush=True)
    with open(ARQUIVO_LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def main() -> None:
    # As chaves vivem no .env da raiz do projeto, fora do repositorio.
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), ".env"))

    ap = argparse.ArgumentParser(prog="v9-vivo")
    ap.add_argument("--armar", action="store_true",
                    help="envia ordens de verdade (sem isto, so mostra)")
    ap.add_argument("--real", action="store_true",
                    help="opera na conta REAL em vez da testnet")
    ap.add_argument("--aceitar-defasagem", dest="aceitar_defasagem",
                    action="store_true",
                    help="opera mesmo com dados velhos (NAO recomendado)")
    a = ap.parse_args()

    if a.real and a.armar:
        registrar("ATENCAO: modo REAL solicitado.")
        resposta = input("Digite EU CONFIRMO para operar dinheiro real: ")
        if resposta.strip() != "EU CONFIRMO":
            registrar("Cancelado pelo operador.")
            return

    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]

    registrar("=" * 62)
    registrar(f"SOBREPOSICAO V9 — {len(ativos)} ativos")

    # 1. Corretora primeiro: e dela que vem a atualizacao dos dados.
    carteira = CarteiraBinance(ativos, testnet=not a.real, armado=a.armar,
                               log=registrar)
    carteira.conectar()

    # 2. Historico local + o que aconteceu desde que ele foi gravado.
    registrar("Conferindo defasagem dos dados:")
    fonte = FonteParquet(cfg=cfg)
    fonte.carregar(ativos)
    historicos, em_dia = atualizar(carteira.api, fonte, ativos, cfg,
                                   log=registrar)

    if not em_dia and not a.aceitar_defasagem:
        registrar("")
        registrar("ABORTADO: os dados nao estao em dia e a decisao seria")
        registrar("tomada sobre um mercado que ja mudou. Rode com")
        registrar("--aceitar-defasagem se souber o que esta fazendo.")
        sys.exit(2)

    oraculo = OraculoTreinado(historicos, CAMINHO_PADRAO)

    ref = ativos[0]
    visao = historicos[ref].em(len(historicos[ref]) - 1)
    regime = oraculo.regime(visao)
    prob = oraculo.probabilidade(visao)
    exposto = regime is not Regime.BEAR

    registrar(f"Dados ate {visao.ts:%Y-%m-%d %H:%M} | modelo treinado em "
              f"{oraculo.meta['treinado_em']}")
    registrar(f"Regime: {regime.value} (prob. de alta {prob:.3f}) "
              f"-> alvo {'EXPOSTO' if exposto else 'CAIXA'}")

    if abs(prob - 0.5) < 0.02:
        registrar("  (o modelo esta praticamente indeciso neste bloco)")

    # 3. Executa.
    carteira.configurar_alavancagem()
    ok = carteira.ajustar(exposto)
    registrar("Ciclo concluido." if ok
              else "Ciclo concluido COM FALHAS — verifique o log acima.")
    registrar("=" * 62)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
