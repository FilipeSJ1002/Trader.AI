# 📈 TRADER.AI

> **Sistema de Trading Algorítmico Autônomo Baseado em Análise Técnica Quantitativa.**
>
> *Projeto de Trabalho de Conclusão de Curso (TCC) - Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-Etapa%201%20(MVP)-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Visão Geral do Projeto

O **TRADER.AI** é uma solução de software desenvolvida para automatizar o processo de tomada de decisão no mercado de criptoativos, com foco específico no par **BTC/USDT**.

Diferente de bots tradicionais baseados em regras simples, este sistema utiliza uma arquitetura modular que visa integrar **Estratégias de Confluência** (união de múltiplos indicadores técnicos) e, futuramente, modelos de **Machine Learning** para mitigar o viés emocional humano e explorar ineficiências de mercado em operações 24/7.

Atualmente, o projeto encontra-se na **Fase 1 (MVP - Minimum Viable Product)**, consistindo em uma API RESTful capaz de processar dados de mercado e retornar decisões de trading baseadas em lógica matemática rigorosa.

## 🚀 Funcionalidades da Etapa 1

Nesta primeira etapa, o sistema foca na validação da arquitetura e da lógica de análise técnica:

* **API de Alta Performance:** Construída com `FastAPI` para garantir baixa latência no processamento de requisições.
* **Motor de Análise Técnica (Core):** Implementação robusta de indicadores utilizando a biblioteca `pandas-ta`.
* **Estratégia de Confluência:** O algoritmo de decisão não depende de um único sinal. Ele exige a confirmação mútua entre:
    * 📉 **RSI (IFR):** Para detectar condições de sobrecompra/sobrevenda.
    * 📊 **Bandas de Bollinger:** Para medir a volatilidade e identificar rompimentos.
    * 📈 **MACD:** Para confirmar a direção e força da tendência.
    * 📏 **Médias Móveis (EMA 50/200):** Para análise de tendência de longo prazo.
* **Simulação de Mercado (Mock Data):** Módulo gerador de dados estocásticos que simula movimentos realistas do Bitcoin para validação segura dos algoritmos.

## 📂 Estrutura do Projeto

A organização do código segue os padrões de *Clean Code* e modularidade:

```text
Trader.AI/
├── main.py            # Ponto de entrada da API (Rotas e Configuração do Servidor)
├── strategy.py        # Lógica de Negócios: Cálculos dos indicadores e Regras de Trade
├── mock_data.py       # (Temp) Gerador de dados fictícios para testes da Etapa 1
├── requirements.txt   # Lista de dependências do projeto
├── README.md          # Documentação oficial
└── .gitignore         # Arquivos ignorados pelo controle de versão
```

##

## 💻 Instalação e Execução
Siga os passos abaixo para rodar o projeto em sua máquina local.

### 1. Pré-requisitos
* Python 3.10 ou superior instalado.
* Git instalado.

### 2. Clonar o Repositório
```Code
git clone [https://github.com/SEU_USUARIO/Trader.AI.git](https://github.com/SEU_USUARIO/Trader.AI.git)
cd Trader.AI
```

### 3. Configurar o Ambiente Virtual (Recomendado)
Isolar as dependências é uma boa prática de desenvolvimento Python.

No Windows:
```Code
python -m venv venv
.\venv\Scripts\activate
```
No Linux ou macOS:
```Code
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependências
```Code
pip install -r requirements.txt
```

### 5. Iniciar o Servidor
Execute o comando abaixo para iniciar a API. O parâmetro --reload permite que o servidor reinicie automaticamente ao salvar alterações no código.
```Code
python -m uvicorn main:app --reload
```
✅ Sucesso: O servidor estará rodando em http://127.0.0.1:8000.

##

## 🧪 Como Realizar Testes
O sistema funciona recebendo um "preço atual" e retornando uma análise completa baseada no histórico. Você pode testar usando o Insomnia, Postman ou a documentação automática do Swagger.

Exemplo de Requisição (Input)
Faça uma requisição do tipo POST para o endpoint /analisar_mercado.

```text
URL: http://127.0.0.1:8000/analisar_mercado Header: Content-Type: application/json
```

Corpo (JSON): Imagine que o Bitcoin acabou de ter um pico de preço. Enviamos esse dado para a API:
```Code
{
  "price": 75000.00,
  "volume": 2500.0
}
```

Exemplo de Resposta (Output)
A API processará esse preço, adicionará ao histórico, recalculará todos os indicadores e retornará a decisão:
```Code
{
	"decision": "VENDA FORTE",
	"analysis": {
		"rsi": 78.5,
		"bollinger_position": "UPPER_BAND_BREAKOUT",
		"macd_signal": "BEARISH_CROSSOVER",
		"trend": "BULLISH (Price > EMA200)"
	},
	"timestamp": "2026-01-05T21:45:00",
	"message": "O ativo está sobrecomprado (RSI > 70) e rompeu a Banda Superior. Probabilidade alta de correção."
}
```

##

## 📅 Roadmap de Desenvolvimento
O desenvolvimento do TRADER.AI segue um cronograma incremental:

* [x]  Etapa 1: Arquitetura Base, API e Estratégia de Confluência (Mock Data).

- [ ]  Etapa 2: Integração com Dataset Histórico Real (Backtesting com dados CSV/Parquet).

* [ ]  Etapa 3: Conexão com API Binance (Leitura de mercado em Tempo Real).

- [ ]  Etapa 4: Execução de Ordens (Live Trading e Gestão de Risco Automatizada).

## ✒️ Autor
Desenvolvido por **Filipe Spirlandeli Junqueira**.

##

```text
Este projeto é estritamente educacional e experimental. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em contas reais.
```
