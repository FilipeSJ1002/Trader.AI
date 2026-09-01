# -*- coding: utf-8 -*-
"""
app/cli.py — a entrega da Sprint 1: o TETO do projeto
======================================================

Roda o oraculo perfeito, o chao (moeda) e os controles fixos, cada um em todas
as fases do ciclo, e imprime a tabela com margem de erro.

O criterio de decisao da Sprint 2 esta no fim da saida, e e o motivo de tudo
isto existir: se o oraculo PERFEITO nao superar com folga o zero, o
comprar-e-segurar e a moeda, entao nenhum classificador treinado vale a pena —
porque o perfeito ja nao bastaria, e nenhum modelo chega perto do perfeito.

Uso:
  python -m app.cli teto
  python -m app.cli teto --de 2021-01-01 --ate 2026-07-25
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

from avaliacao.metricas import comprar_e_segurar
from avaliacao.robustez import limiar_corrigido, tabela, varrer_fases
from dados.fonte import FonteParquet, ler_config
from execucao.papel import CorretoraPapel
from execucao.risco import Risco
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.tipos import Regime
from oraculo.teto import OraculoFixo, OraculoMoeda, OraculoPerfeito


def comando_teto(a: argparse.Namespace) -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    de, ate = datetime.fromisoformat(a.de), datetime.fromisoformat(a.ate)

    print(f"\nCarregando {len(ativos)} ativos...", flush=True)
    t0 = time.time()
    historicos = FonteParquet(cfg=cfg).carregar(ativos)
    print(f"  pronto em {time.time()-t0:.1f}s | {de:%Y-%m-%d} a {ate:%Y-%m-%d}")

    perfeito = OraculoPerfeito(historicos, limiar_fora=a.limiar_fora)
    print(f"  oraculo perfeito: {perfeito.dias_mapeados:,} dias mapeados "
          f"| {perfeito.distribuicao()}")

    def montar(fabrica_oraculo):
        def _montar():
            return (
                historicos,
                {Regime.BULL: MotorBull(forca_min=a.forca_min),
                 Regime.BEAR: MotorBear(forca_min=a.forca_min)},
                fabrica_oraculo(),
                CorretoraPapel(saldo_inicial=a.capital),
                Risco(fracao_por_operacao=a.fracao,
                      alavancagem=a.alavancagem,
                      atr_stop=a.atr_stop, atr_alvo=a.atr_alvo,
                      max_posicoes=a.max_posicoes, prazo_atr=a.prazo_atr),
                ativos[0],
            )
        return _montar

    cenarios = [
        ("TETO — oraculo perfeito", lambda: perfeito),
        ("CHAO — moeda", lambda: OraculoMoeda(semente=7)),
        ("controle — sempre BULL", lambda: OraculoFixo(Regime.BULL)),
        ("controle — sempre BEAR", lambda: OraculoFixo(Regime.BEAR)),
    ]

    resultados = []
    for nome, fabrica in cenarios:
        print(f"\n  {nome} ...", end="", flush=True)
        t0 = time.time()
        r, _ = varrer_fases(nome, montar(fabrica), de, ate,
                            a_cada=a.ciclo,
                            aoprogresso=lambda k, n: print(".", end="",
                                                           flush=True))
        resultados.append(r)
        print(f" {r.media*100:+.2f}% ± {r.erro*100:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    ref = comprar_e_segurar(historicos, de, ate, a.capital)
    print(tabela(resultados, ref, titulo=f"TETO DO PROJETO — {de:%Y-%m-%d} a "
                                         f"{ate:%Y-%m-%d}"))

    _veredito(resultados, ref, len(cenarios))


def _veredito(resultados, ref, n_comparacoes: int) -> None:
    """O criterio de decisao da Sprint 2, aplicado aos numeros que sairam."""
    por_nome = {r.nome: r for r in resultados}
    teto = por_nome["TETO — oraculo perfeito"]
    chao = por_nome["CHAO — moeda"]

    print("\n" + "=" * 96)
    print("  CRITERIO DE DECISAO DA SPRINT 2")
    print("=" * 96)
    print(f"\n  Limiar corrigido para {n_comparacoes} comparacoes: "
          f"p < {limiar_corrigido(n_comparacoes):.4f} (|t| ~ 3,5 ou mais)\n")

    checagens = [
        ("o teto e distinguivel de zero",
         abs(teto.t) >= 2.0, f"t = {teto.t:+.2f}"),
        ("o teto supera a moeda",
         teto.media > chao.media + 2 * (teto.erro + chao.erro),
         f"{teto.media*100:+.2f}% vs {chao.media*100:+.2f}%"),
        ("o teto supera comprar-e-segurar",
         teto.media > ref.retorno,
         f"{teto.media*100:+.2f}% vs {ref.retorno*100:+.2f}%"),
    ]
    for descricao, passou, detalhe in checagens:
        print(f"    [{'SIM' if passou else 'NAO'}] {descricao:<38} {detalhe}")

    aprovado = all(p for _, p, _ in checagens)
    print()
    if aprovado:
        print("  CONSTRUIR O ORACULO REAL. O teto tem espaco, e a distancia ate")
        print("  a moeda e o premio que um classificador de regime disputa.")
    else:
        print("  NAO CONSTRUIR O ORACULO REAL.")
        print("  O oraculo PERFEITO ja nao basta — e nenhum modelo treinado")
        print("  chega perto do perfeito. Gastar meses treinando um")
        print("  classificador para este alvo seria mirar num alvo que nao esta la.")
        print("  O que muda o quadro nao e o modelo: sao os motores, o custo")
        print("  por operacao ou a escala de tempo.")
    print("=" * 96)


def main() -> None:
    ap = argparse.ArgumentParser(prog="trader-v9")
    sub = ap.add_subparsers(dest="comando", required=True)

    t = sub.add_parser("teto", help="mede o teto do projeto")
    t.add_argument("--de", default="2021-01-01")
    t.add_argument("--ate", default="2026-07-25")
    t.add_argument("--capital", type=float, default=5000.0)
    t.add_argument("--ciclo", type=int, default=15,
                   help="minutos entre avaliacoes (e o numero de fases)")
    t.add_argument("--limiar-fora", dest="limiar_fora", type=float, default=0.0,
                   help="movimento diario minimo para valer operar")
    t.add_argument("--forca-min", dest="forca_min", type=float, default=0.35)
    t.add_argument("--fracao", type=float, default=0.20)
    t.add_argument("--alavancagem", type=float, default=1.0)
    t.add_argument("--atr-stop", dest="atr_stop", type=float, default=1.5)
    t.add_argument("--atr-alvo", dest="atr_alvo", type=float, default=3.0)
    t.add_argument("--max-posicoes", dest="max_posicoes", type=int, default=3)
    t.add_argument("--prazo-atr", dest="prazo_atr", default="diario",
                   choices=["minuto", "h4", "diario"])
    t.set_defaults(func=comando_teto)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
