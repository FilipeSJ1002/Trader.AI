# -*- coding: utf-8 -*-
"""
v6_diagnostico.py — Trader.AI: varredura completa de erros do projeto
======================================================================

Roda todas as verificacoes de uma vez e gera um relatorio unico com tudo
que esta quebrado ou suspeito. Feito para colar o resultado no chat.

Verifica:
  1. SINTAXE      — todo .py compila?
  2. IMPORTS      — todo modulo importa sem erro?
  3. ARQUIVOS     — modelos, dados e .env estao no lugar?
  4. DEPENDENCIAS — pacotes do requirements instalados?
  5. LINTER       — erros de tipo/nome (se pyright ou ruff estiverem disponiveis)
  6. INTEGRIDADE  — funcoes-chave importaveis e coerentes entre modulos
  7. PROCESSOS    — o que esta rodando agora

Uso:
  python v6_diagnostico.py              # relatorio no console
  python v6_diagnostico.py --salvar     # + salva em relatorios/diagnostico.txt
"""
import os
import re
import sys
import glob
import json
import argparse
import importlib
import subprocess
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
REL  = os.path.join(BASE, "relatorios")
os.makedirs(REL, exist_ok=True)

linhas = []
problemas = []


def out(msg=""):
    print(msg, flush=True)
    linhas.append(msg)


def secao(titulo):
    out("")
    out("=" * 74)
    out(f"  {titulo}")
    out("=" * 74)


def problema(categoria, arquivo, detalhe):
    problemas.append((categoria, arquivo, detalhe))


# ── 1. SINTAXE ──────────────────────────────────────────────────────────────
def checar_sintaxe():
    secao("1. SINTAXE — todo arquivo .py compila?")
    import py_compile
    arquivos = sorted(glob.glob(os.path.join(BASE, "*.py")))
    ok = 0
    for f in arquivos:
        nome = os.path.basename(f)
        try:
            py_compile.compile(f, doraise=True, quiet=1)
            ok += 1
        except py_compile.PyCompileError as e:
            out(f"  [ERRO] {nome}")
            out(f"         {str(e).splitlines()[-1][:120]}")
            problema("SINTAXE", nome, str(e).splitlines()[-1][:200])
        except Exception as e:
            out(f"  [ERRO] {nome}: {type(e).__name__}: {e}")
            problema("SINTAXE", nome, f"{type(e).__name__}: {e}")
    out(f"  {ok}/{len(arquivos)} arquivos compilam sem erro")


# ── 2. IMPORTS ──────────────────────────────────────────────────────────────
def checar_imports():
    secao("2. IMPORTS — cada modulo carrega sem erro?")
    # Modulos que nao devem ser importados (executam algo ao carregar)
    pular = {"main", "v6_diagnostico"}
    arquivos = sorted(glob.glob(os.path.join(BASE, "*.py")))
    sys.path.insert(0, BASE)
    ok = 0
    for f in arquivos:
        nome = os.path.basename(f)[:-3]
        if nome in pular:
            continue
        try:
            importlib.import_module(nome)
            ok += 1
        except SystemExit as e:
            # Scripts de verificacao chamam sys.exit() ao rodar; sem este
            # catch, um deles derruba o diagnostico inteiro.
            msg = f"o script chamou sys.exit({e.code}) durante a importacao"
            out(f"  [ERRO] {nome}.py -> {msg}")
            problema("IMPORT", nome + ".py", msg)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:110]}"
            out(f"  [ERRO] {nome}.py -> {msg}")
            problema("IMPORT", nome + ".py", msg)
    out(f"  {ok} modulos importam corretamente")


# ── 3. ARQUIVOS ESSENCIAIS ──────────────────────────────────────────────────
def checar_arquivos():
    secao("3. ARQUIVOS — o que existe e o que falta")
    itens = [
        ("v5_model_b.pth",  "modelo em producao",       True),
        ("v6_model.pth",    "modelo V6 (em treino)",    False),
        (".env",            "chaves de API",            True),
        ("data",            "dados historicos",         True),
        ("relatorios",      "saidas de execucao",       False),
        ("v5_live_state.json", "estado do paper trading", False),
    ]
    for caminho, desc, obrigatorio in itens:
        p = os.path.join(BASE, caminho)
        if os.path.exists(p):
            if os.path.isdir(p):
                n = len(os.listdir(p))
                out(f"  [OK]   {caminho:<24} {desc} ({n} arquivos)")
            else:
                mb = os.path.getsize(p) / 1024 / 1024
                out(f"  [OK]   {caminho:<24} {desc} ({mb:.1f} MB)")
        else:
            nivel = "FALTA" if obrigatorio else "aviso"
            out(f"  [{nivel}] {caminho:<24} {desc}")
            if obrigatorio:
                problema("ARQUIVO", caminho, f"{desc} nao encontrado")

    # Chaves no .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        for k in ["BINANCE_API_KEY", "BINANCE_FUTURES_API_KEY"]:
            v = os.getenv(k)
            out(f"  [{'OK' if v else 'FALTA'}]   {k:<24} "
                f"{'definida' if v else 'AUSENTE no .env'}")
            if not v and "FUTURES" in k:
                problema("CONFIG", ".env", f"{k} ausente (necessaria p/ executor)")
    except Exception as e:
        out(f"  [aviso] nao foi possivel ler o .env: {e}")


# ── 4. DEPENDENCIAS ─────────────────────────────────────────────────────────
def checar_dependencias():
    secao("4. DEPENDENCIAS — pacotes essenciais instalados?")
    pacotes = ["torch", "pandas", "numpy", "pandas_ta", "binance",
               "requests", "dotenv", "pyarrow", "fastapi", "sklearn"]
    for p in pacotes:
        try:
            m = importlib.import_module(p)
            v = getattr(m, "__version__", "?")
            out(f"  [OK]   {p:<14} {v}")
        except ImportError:
            out(f"  [FALTA] {p:<14} nao instalado")
            problema("DEPENDENCIA", p, "pacote nao instalado")


# ── 5. LINTER ───────────────────────────────────────────────────────────────
def checar_linter():
    secao("5. LINTER — os mesmos erros que o VSCode mostra (pyright)")
    # O VSCode usa Pylance, que e o pyright por baixo. Rodar o pyright aqui
    # reproduz exatamente os sublinhados vermelhos do editor. Se ele nao
    # estiver instalado, o npx baixa sob demanda (precisa de Node + internet).
    npx = "npx.cmd" if os.name == "nt" else "npx"   # no Windows npx e um .cmd
    candidatos = [(["pyright", "--outputjson"], "pyright"),
                  ([npx, "-y", "pyright@latest", "--outputjson"], "pyright (npx)")]

    for cmd, nome in candidatos:
        try:
            r = subprocess.run(cmd, cwd=BASE, capture_output=True,
                               text=True, timeout=600)
        except FileNotFoundError:
            continue
        except Exception as e:
            out(f"  [aviso] {nome} falhou: {e}")
            continue

        try:
            dados = json.loads(r.stdout)
        except Exception:
            out(f"  [aviso] {nome} rodou mas nao devolveu JSON valido")
            for l in ((r.stdout or "") + (r.stderr or "")).splitlines()[:10]:
                out(f"    {l[:130]}")
            return

        diags = [d for d in dados.get("generalDiagnostics", [])
                 if d.get("severity") == "error"]
        n_arq = dados.get("summary", {}).get("filesAnalyzed", "?")
        if not diags:
            out(f"  [OK] {nome}: 0 erros em {n_arq} arquivos analisados")
            return

        out(f"  [{nome}] {len(diags)} erro(s) em {n_arq} arquivos:")
        for d in diags[:40]:
            arq  = os.path.basename(d.get("file", "?"))
            lin  = d.get("range", {}).get("start", {}).get("line", 0) + 1
            regra = d.get("rule", "-")
            msg  = " ".join(str(d.get("message", "")).split())[:100]
            out(f"    {arq}:{lin} [{regra}] {msg}")
            problema("LINTER", f"{arq}:{lin}", f"[{regra}] {msg}")
        if len(diags) > 40:
            out(f"    ... e mais {len(diags)-40} erro(s)")
        return

    out("  (pyright nao encontrado — instale Node.js ou rode: pip install pyright)")
    out("  Regras silenciadas de proposito ficam em pyrightconfig.json")


# ── 6. INTEGRIDADE ENTRE MODULOS ────────────────────────────────────────────
def checar_integridade():
    secao("6. INTEGRIDADE — a estrategia e a mesma em todos os modulos?")
    try:
        sys.path.insert(0, BASE)
        import v5_backtest, v6_ciclo, v5_live
        checagens = [
            ("v1_scores identico (ciclo vs backtest)",
             v6_ciclo.v1_scores is v5_backtest.v1_scores),
            ("leverage_for identico (ciclo vs backtest)",
             v6_ciclo.leverage_for is v5_backtest.leverage_for),
            ("v1_scores identico (live vs backtest)",
             v5_live.v1_scores is v5_backtest.v1_scores),
            ("SL do ciclo == SL do backtest (0.005)",
             abs(v6_ciclo.SL_PCT - 0.005) < 1e-9),
            ("TP do ciclo == TP do backtest (0.010)",
             abs(v6_ciclo.TP_PCT - 0.010) < 1e-9),
            ("limiar de confianca == 0.52",
             abs(v6_ciclo.DIRCONF_MIN - 0.52) < 1e-9),
        ]
        for desc, ok in checagens:
            out(f"  [{'OK' if ok else 'FALHA'}] {desc}")
            if not ok:
                problema("INTEGRIDADE", "v6_ciclo.py", desc)
    except Exception as e:
        out(f"  [ERRO] nao foi possivel verificar: {type(e).__name__}: {e}")
        problema("INTEGRIDADE", "-", str(e)[:150])


# ── 7. PROCESSOS ────────────────────────────────────────────────────────────
def checar_processos():
    secao("7. PROCESSOS — o que esta rodando agora")
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                           capture_output=True, text=True, timeout=30)
        n = r.stdout.lower().count("python.exe")
        out(f"  {n} processo(s) python em execucao")
        out("  (lembre: o venv duplica cada script — 2 processos por script e normal)")
    except Exception:
        out("  (nao foi possivel listar processos)")

    # ATENCAO: cada script escreve o log num lugar diferente — o treino e o
    # paper trading gravam na RAIZ do projeto; so o executor grava em
    # relatorios/. Procurar no lugar errado ja fez uma sessao inteira concluir
    # que o treino estava travado quando ele estava rodando normalmente.
    import time
    for log_nome, desc in [("v6_training.log", "treino V6"),
                           ("v5_live.log", "paper trading"),
                           ("v6_executor.log", "executor")]:
        candidatos = [os.path.join(BASE, log_nome), os.path.join(REL, log_nome)]
        existentes = [c for c in candidatos if os.path.exists(c)]
        if not existentes:
            out(f"  {desc:<16} (sem log)")
            continue
        # o mais recente e o que vale; os outros sao copias antigas
        p = max(existentes, key=os.path.getmtime)
        idade = (time.time() - os.path.getmtime(p)) / 60
        with open(p, encoding="utf-8", errors="replace") as f:
            linhas_log = [l for l in f.readlines() if l.strip()]
        ultima = linhas_log[-1].strip()[:100] if linhas_log else "(vazio)"
        rel_p = os.path.relpath(p, BASE)
        out(f"  {desc:<16} {rel_p:<28} ha {idade:>5.0f} min | {ultima}")
        if len(existentes) > 1:
            antigo = os.path.relpath(min(existentes, key=os.path.getmtime), BASE)
            out(f"  {'':<16} (aviso: existe copia desatualizada em {antigo})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salvar", action="store_true",
                    help="Salva o relatorio em relatorios/diagnostico.txt")
    a = ap.parse_args()

    out("=" * 74)
    out(f"  TRADER.AI — DIAGNOSTICO COMPLETO")
    out(f"  {datetime.now():%Y-%m-%d %H:%M:%S} | Python {sys.version.split()[0]}")
    out("=" * 74)

    checar_sintaxe()
    checar_imports()
    checar_arquivos()
    checar_dependencias()
    checar_integridade()
    checar_linter()
    checar_processos()

    secao("RESUMO")
    if not problemas:
        out("  Nenhum problema encontrado. Projeto integro.")
    else:
        out(f"  {len(problemas)} problema(s) encontrado(s):")
        out("")
        por_cat = {}
        for cat, arq, det in problemas:
            por_cat.setdefault(cat, []).append((arq, det))
        for cat, itens in por_cat.items():
            out(f"  [{cat}] {len(itens)} ocorrencia(s):")
            for arq, det in itens:
                out(f"     - {arq}: {det}")
        out("")
        out("  >> Copie este relatorio e envie no chat para correcao.")
    out("=" * 74)

    if a.salvar:
        destino = os.path.join(REL, "diagnostico.txt")
        with open(destino, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))
        print(f"\nRelatorio salvo em: {destino}")


if __name__ == "__main__":
    main()
