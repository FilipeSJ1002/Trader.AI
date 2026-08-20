# -*- coding: utf-8 -*-
"""
v7_edge_funding.py — o funding rate tem edge? (a porta que faltava)
====================================================================

Hipotese: o funding rate mede o desequilibrio de alavancagem entre comprados
e vendidos. Quando fica extremo, um lado esta pagando caro para manter a
posicao — sinal classico de excesso, que costuma preceder reversao.

E informacao que o PRECO NAO CONTEM: e o comportamento dos participantes.

Duas fontes de retorno, medidas separadamente:

  1. PRECO      — apostar contra a multidao funciona?
  2. CARRY      — quem fica do lado que RECEBE funding ganha a cada 8h,
                  independentemente do preco. Esta e a parte mais confiavel
                  e a mais ignorada.

Metodologia identica ao resto do projeto: retorno no sentido da aposta,
custo descontado, erro-padrao, e robustez por ativo e por ano. Sem isso,
qualquer numero bonito aqui e so mais uma ilusao.

Uso:
  python v7_edge_funding.py                    # contra a multidao (reversao)
  python v7_edge_funding.py --sentido momentum # a favor, para comparar
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from v5_data_prep import _load_parquet
from v5_backtest import FEE

DEST = "data_alt"
HORAS = [8, 24, 48, 72, 168]      # horizontes em horas


def resumo_horizonte(nome, dados, n):
    print(f"\n  {nome}  (n = {n})")
    print(f"  {'horizonte':>10} {'preco':>10} {'+carry':>10} {'erro-padrao':>13} "
          f"{'t':>7}  veredito")
    print("  " + "-" * 68)
    for h in HORAS:
        r = dados[h]["preco"]
        rc = dados[h]["total"]
        if len(r) < 30:
            print(f"  {h:>7}h   (poucas)")
            continue
        m, mc = r.mean(), rc.mean()
        se = rc.std(ddof=1) / np.sqrt(len(rc))
        t = mc / se if se > 0 else 0.0
        vered = ("SINAL POSITIVO" if t >= 2 else
                 "negativo" if t <= -2 else "dentro do ruido")
        print(f"  {h:>7}h {m*100:>+9.4f}% {mc*100:>+9.4f}% "
              f"{'+/- ' + format(se*100, '.4f') + '%':>13} {t:>+7.2f}  {vered}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentido", choices=["reversao", "momentum"],
                    default="reversao",
                    help="reversao = aposta CONTRA o lado que paga funding")
    ap.add_argument("--z", type=float, default=1.5,
                    help="quantos desvios para considerar 'extremo' (padrao 1,5)")
    ap.add_argument("--janela", type=int, default=90,
                    help="periodos de funding para o z-score (90 = 30 dias)")
    ap.add_argument("--detalhe", action="store_true",
                    help="abre por ativo e por ano")
    ap.add_argument("--sem-sobreposicao", dest="sem_sobrep", type=int, default=0,
                    metavar="HORAS",
                    help="so aceita um evento a cada N horas por ativo. "
                         "OBRIGATORIO para horizontes longos: eventos a cada 8h "
                         "com posicao de 1 semana geram 21 janelas sobrepostas, "
                         "o que infla a significancia em ~4,6x.")
    a = ap.parse_args()

    taxa = FEE * 2
    arquivos = sorted(glob.glob(os.path.join(DEST, "funding_*.parquet")))
    if not arquivos:
        raise SystemExit("Rode antes: python v7_coleta_alternativa.py --funding")

    linhas = []
    print(f"\nMedindo funding | sentido: {a.sentido} | extremo: |z| >= {a.z} "
          f"| janela z: {a.janela} periodos")

    for arq in arquivos:
        sym = os.path.basename(arq).replace("funding_", "").replace(".parquet", "")
        if not os.path.exists(f"data/{sym}_1m.parquet"):
            continue
        fund = pd.read_parquet(arq)["fundingRate"]
        preco = _load_parquet(sym)["close"]

        # z-score do funding: 'extremo' e relativo ao proprio historico recente
        z = ((fund - fund.rolling(a.janela).mean())
             / (fund.rolling(a.janela).std() + 1e-12))

        n_sym = 0
        ultimo_aceito = None
        for ts, zi in z.dropna().items():
            if abs(zi) < a.z:
                continue
            # Janelas independentes: descarta eventos que se sobrepoem ao anterior
            if a.sem_sobrep:
                if ultimo_aceito is not None and \
                   (ts - ultimo_aceito) < pd.Timedelta(hours=a.sem_sobrep):
                    continue
                ultimo_aceito = ts
            # funding POSITIVO = comprados pagam vendidos = excesso de comprados
            # reversao -> vender ; momentum -> comprar
            if a.sentido == "reversao":
                direcao = -1.0 if zi > 0 else 1.0
            else:
                direcao = 1.0 if zi > 0 else -1.0

            try:
                p0 = float(preco.asof(ts))
            except Exception:
                continue
            if not np.isfinite(p0) or p0 <= 0:
                continue

            reg = {"sym": sym, "ano": ts.year, "z": float(zi)}
            ok = True
            for h in HORAS:
                try:
                    p1 = float(preco.asof(ts + pd.Timedelta(hours=h)))
                except Exception:
                    ok = False
                    break
                if not np.isfinite(p1) or p1 <= 0:
                    ok = False
                    break
                ret_preco = direcao * (p1 / p0 - 1)
                # CARRY: com funding POSITIVO, comprados pagam e vendidos
                # recebem, a cada 8h. Logo, para direcao d:
                #   d = -1 (vendido) e f > 0  ->  recebe  (+f)
                #   d = +1 (comprado) e f > 0 ->  paga    (-f)
                # ou seja: carry = -d * f * periodos
                periodos = h // 8
                carry = -direcao * float(fund.loc[ts]) * periodos
                reg[f"p{h}"] = ret_preco - taxa
                reg[f"t{h}"] = ret_preco + carry - taxa
            if ok:
                linhas.append(reg)
                n_sym += 1

        print(f"  {sym}: {n_sym} eventos extremos", flush=True)

    if not linhas:
        print("Nenhum evento extremo no periodo.")
        return

    df = pd.DataFrame(linhas)
    dados = {h: {"preco": df[f"p{h}"].values, "total": df[f"t{h}"].values}
             for h in HORAS}

    print(f"\n{'='*78}")
    print(f"  EDGE DO FUNDING RATE — aposta por {a.sentido.upper()} | |z| >= {a.z}")
    print(f"  'preco' = so o movimento | '+carry' = movimento + funding recebido")
    print(f"  Custo descontado: {taxa*100:.3f}% por operacao")
    print(f"{'='*78}")
    resumo_horizonte("TODOS OS EVENTOS", dados, len(df))

    if a.detalhe:
        melhor = max(HORAS, key=lambda h: df[f"t{h}"].mean())
        col = f"t{melhor}"
        print(f"\n{'='*78}")
        print(f"  ROBUSTEZ NO MELHOR HORIZONTE ({melhor}h) — com carry")
        print(f"{'='*78}")
        for titulo, chave in (("POR ATIVO", "sym"), ("POR ANO", "ano")):
            print(f"\n  {titulo}")
            print(f"  {'':<12} {'n':>6} {'media':>11} {'erro-padrao':>13} {'t':>7}")
            print("  " + "-" * 54)
            pos = 0
            grupos = sorted(df[chave].unique())
            for g in grupos:
                sub = df[df[chave] == g][col]
                if len(sub) < 15:
                    print(f"  {str(g):<12} {len(sub):>6}  (amostra pequena)")
                    continue
                se = sub.std(ddof=1) / np.sqrt(len(sub))
                t = sub.mean() / se if se > 0 else 0.0
                if sub.mean() > 0:
                    pos += 1
                print(f"  {str(g):<12} {len(sub):>6} {sub.mean()*100:>+10.4f}% "
                      f"{'+/- ' + format(se*100, '.4f') + '%':>13} {t:>+7.2f}")
            print(f"  -> {pos} de {len(grupos)} com media positiva")

    print(f"\n{'='*78}")
    print("  COMO LER")
    print("    |t| >= 2 no '+carry' = o sinal nao e ruido.")
    print("    Um edge real aparece em varios ativos E varios anos.")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
