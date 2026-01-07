# Classificação de Dias de Alto Consumo Energético em uma ETE de Melbourne

Este repositório contém o projeto do **Homework 3** da disciplina **Inteligência Computacional Aplicada (ICA)**, cujo objetivo é comparar modelos de **classificação binária** (linear e não lineares) para prever dias de **alto consumo energético** (`HighEnergy = 1`) em uma estação de tratamento de esgoto (ETE) de Melbourne, Austrália.

O estudo utiliza variáveis **operacionais e meteorológicas diárias** no período de **2014–2019** e avalia o desempenho de modelos lineares e não lineares, com validação estatística baseada no **Teste de McNemar**.

📄 **Relatório final (formato IEEE, 6 páginas):**
- `report/hw3_classification_high_energy_consumption.pdf`

---

## 📁 Estrutura do repositório
```shell
  $ tree
.
├─ report/
│  └─ hw3_classification_high_energy_consumption.pdf   # relatório final (IEEE)
├─ code/
│  ├─ hw3_prepare_classification.py       # cria HighEnergy + pré-processamento
│  └─ hw3_models_classification.py        # treina LDA, kNN, SVM + McNemar
├─ data/
│  └─ Data-Melbourne_F_clean.csv          # dataset limpo (HW2)
├─ figures/
│  └─ model_comparison_bar.png            # comparação visual dos modelos
├─ outputs_hw3/
│  ├─ train_classification.csv            # treino (1015 obs) + HighEnergy
│  ├─ test_classification.csv             # teste (339 obs) + HighEnergy
│  └─ hw3_classification_summary.csv      # TN/FP/FN/TP (tabelas LaTeX)
└─ README.md
```
---


#Dependências

Principais dependências em Python:

Python ≥ 3.9

numpy

pandas

scikit-learn

matplotlib

seaborn

mlxtend (Teste de McNemar)

Instalação:

pip install numpy pandas scikit-learn matplotlib seaborn mlxtend

▶️ Como executar o código (reprodutibilidade total)
###1. Criar ambiente virtual (recomendado)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

2. Instalar dependências
pip install numpy pandas scikit-learn matplotlib seaborn mlxtend

3. Preparar os dados de classificação
python code/hw3_prepare_classification.py


Este script:

- Carrega data/Data-Melbourne_F_clean.csv (HW2);

- Cria HighEnergy = 1 se total_grid > 275808 kWh/dia (mediana);

- Divide treino (75%, 1015 obs) e teste (25%, 339 obs), de forma estratificada;

- Aplica transformação log(1+x) em PP e padronização z-score;

- Salva os arquivos de treino e teste em outputs_hw3/.

###4. Treinar modelos e gerar resultados
python code/hw3_models_classification.py


Este script:

- Treina os modelos LDA, k-NN e SVM-RBF;

- Otimiza hiperparâmetros com GridSearchCV e validação cruzada 5-fold;

- Calcula matrizes de confusão e métricas de desempenho;

- Gera o gráfico comparativo em figures/model_comparison_bar.png;

- Executa o Teste de McNemar (k-NN vs LDA);

- Salva hw3_classification_summary.csv para uso direto no LaTeX.

###5. Compilar o relatório (opcional)
cd report
pdflatex hw3_classification_high_energy_consumption.tex
bibtex hw3_classification_high_energy_consumption
pdflatex hw3_classification_high_energy_consumption.tex
pdflatex hw3_classification_high_energy_consumption.tex

## 📊 Principais Resultados (resumo)
| Modelo   | Acurácia | Sensibilidade | F1-Score |
|----------|----------|---------------|----------|
| k-NN     | 67,8%    | 79,9%         | 71,8%   |
| SVM-RBF  | 63,1%    | 72,8%         | 67,2%   |
| LDA      | 59,9%    | 63,3%         | 64,8%   |

📌 Teste de McNemar (k-NN vs LDA):
p-valor = 0.004 → diferença estatisticamente significativa.

##👤 Contribuições

Trabalho individual.

Todas as etapas (limpeza dos dados, implementação dos modelos, análise estatística e elaboração do relatório) foram realizadas por:

Luiz Augusto Gomes da Silva de Jesus

##🤖 Uso de IA

Ferramentas de IA foram utilizadas pontualmente como apoio à revisão de formatação LaTeX e pesquisa bibliográfica. Todas as decisões metodológicas, implementação dos modelos e interpretações dos resultados são de autoria do autor.

##📄 Licença e Contato

Repositório público para fins acadêmicos.

Autor: Luiz Augusto Gomes da Silva de Jesus
Usuário GitHub: luizdevmaster
Disciplina: ICA — Inteligência Computacional Aplicada
Instituição: Universidade Federal do Ceará (UFC)
Ano: 2026
