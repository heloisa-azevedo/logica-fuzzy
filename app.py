"""
Atividade de lógica fuzzy: índice de atratividade de vagas de IA.

Rodar com:  streamlit run app.py
Estrutura esperada:
    app.py
    data/ai_job_dataset.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st
from skfuzzy import control as ctrl

st.set_page_config(page_title="Atratividade de vagas de IA com lógica fuzzy", layout="wide")

CAMINHO_DADOS = Path(__file__).parent / "data" / "ai_job_dataset.csv"
CORES = {"baixa": "#B5523B", "media": "#D9A441", "alta": "#2F6F73"}
NOMES_VAR = {
    "salary": "Salário (USD)",
    "experience": "Experiência exigida (anos)",
    "benefits": "Nota de benefícios",
    "skills_req": "Tecnologias exigidas",
    "attractiveness": "Atratividade",
}
COLUNAS = {
    "salary": "salary_usd",
    "experience": "years_experience",
    "benefits": "benefits_score",
    "skills_req": "skill_count",
}

TECHS = [
    "python", "deep learning", "pytorch", "tensorflow", "aws", "azure", "gcp",
    "docker", "kubernetes", "sql", "nlp", "computer vision", "mlops", "spark",
    "hadoop", "tableau", "data visualization", "statistics", "mathematics",
    "linux", "git", "r", "scala", "java",
]

# ---------------------------------------------------------------------------
# Definição dos dois modelos (original e corrigido) como dados, para que o
# app possa construir, comparar e explicar os dois com o mesmo código.
# ---------------------------------------------------------------------------
SAIDA = {
    "universo": np.arange(0, 101, 1),
    "termos": {
        "baixa": ("tri", [0, 0, 40]),
        "media": ("tri", [30, 50, 70]),
        "alta": ("trap", [60, 80, 100, 100]),
    },
}

REGRAS_ORIGINAIS = [
    ("rule1", [("salary", "alto"), ("benefits", "excelente")], "alta"),
    ("rule2", [("salary", "baixo"), ("experience", "senior")], "baixa"),
    ("rule3", [("salary", "medio"), ("benefits", "bom")], "media"),
    ("rule4", [("salary", "alto"), ("experience", "junior")], "alta"),
    ("rule5", [("salary", "baixo"), ("benefits", "ruim")], "baixa"),
    ("rule6", [("salary", "medio"), ("experience", "pleno")], "media"),
    ("rule7", [("salary", "alto"), ("experience", "senior")], "alta"),
    ("rule_tech_1", [("salary", "alto"), ("skills_req", "muitas")], "alta"),
    ("rule_tech_2", [("salary", "baixo"), ("skills_req", "muitas")], "baixa"),
    ("rule_tech_3", [("salary", "medio"), ("skills_req", "intermediarias")], "media"),
    ("rule_tech_4", [("salary", "alto"), ("skills_req", "poucas")], "alta"),
    ("rule_tech_5", [("salary", "baixo"), ("skills_req", "poucas")], "media"),
]

MODELOS = {
    "corrigido": {
        "contagem": "itens",
        "entradas": {
            "salary": (np.arange(0, 400001, 1000), {
                "baixo": ("trap", [0, 0, 55000, 100000]),
                "medio": ("trap", [60000, 90000, 120000, 160000]),
                "alto": ("trap", [120000, 180000, 400000, 400000]),
            }),
            "experience": (np.arange(0, 21, 1), {
                "junior": ("trap", [0, 0, 1, 4]),
                "pleno": ("tri", [2, 6, 10]),
                "senior": ("trap", [8, 13, 20, 20]),
            }),
            "benefits": (np.linspace(5, 10, 51), {
                "ruim": ("tri", [5, 5, 6.5]),
                "bom": ("tri", [5.8, 7.5, 9.2]),
                "excelente": ("tri", [8.5, 10, 10]),
            }),
            "skills_req": (np.linspace(3, 5, 21), {
                "poucas": ("tri", [3, 3, 4]),
                "intermediarias": ("tri", [3, 4, 5]),
                "muitas": ("tri", [4, 5, 5]),
            }),
        },
        "regras": [
            ("rule_base_1", [("salary", "baixo")], "baixa"),
            ("rule_base_2", [("salary", "medio")], "media"),
            ("rule_base_3", [("salary", "alto")], "alta"),
        ]
        + REGRAS_ORIGINAIS
        + [
            ("rule_new_1", [("salary", "medio"), ("benefits", "excelente")], "alta"),
            ("rule_new_2", [("salary", "medio"), ("benefits", "ruim")], "baixa"),
            ("rule_new_3", [("salary", "medio"), ("skills_req", "poucas")], "alta"),
            ("rule_new_4", [("salary", "medio"), ("skills_req", "muitas")], "baixa"),
            ("rule_new_5", [("salary", "baixo"), ("experience", "junior")], "media"),
            ("rule_new_6", [("salary", "alto"), ("benefits", "ruim")], "media"),
            ("rule_new_7", [("salary", "baixo"), ("benefits", "excelente")], "media"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Funções do modelo
# ---------------------------------------------------------------------------
def curva(universo, tipo, params):
    return fuzz.trimf(universo, params) if tipo == "tri" else fuzz.trapmf(universo, params)


def conta_skills(texto, metodo):
    if pd.isna(texto):
        return 0
    texto = str(texto).lower()
    if metodo == "substring":  # comportamento original, com o bug do 'r'
        return sum(1 for t in TECHS + ["machine learning"] if t in texto)
    itens = {s.strip() for s in texto.split(",") if s.strip()}
    return len(itens & set(TECHS))


@st.cache_resource
def construir_simulacao(versao):
    spec = MODELOS[versao]
    variaveis = {}
    for nome, (universo, termos) in spec["entradas"].items():
        v = ctrl.Antecedent(universo, nome)
        for termo, (tipo, p) in termos.items():
            v[termo] = curva(universo, tipo, p)
        variaveis[nome] = v
    saida = ctrl.Consequent(SAIDA["universo"], "attractiveness")
    for termo, (tipo, p) in SAIDA["termos"].items():
        saida[termo] = curva(SAIDA["universo"], tipo, p)

    regras = []
    for _, antecedentes, consequente in spec["regras"]:
        cond = variaveis[antecedentes[0][0]][antecedentes[0][1]]
        for var, termo in antecedentes[1:]:
            cond = cond & variaveis[var][termo]
        regras.append(ctrl.Rule(cond, saida[consequente]))
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(regras))


def pontuar(simulacao, entradas):
    try:
        for nome, valor in entradas.items():
            simulacao.input[nome] = valor
        simulacao.compute()
        return simulacao.output["attractiveness"]
    except Exception:  # nenhuma regra ativada
        return np.nan


def pertinencias(versao, entradas):
    """Grau de pertinência de cada valor de entrada em cada termo."""
    graus = {}
    for nome, (universo, termos) in MODELOS[versao]["entradas"].items():
        x = np.clip(entradas[nome], universo.min(), universo.max())
        graus[nome] = {
            t: float(fuzz.interp_membership(universo, curva(universo, tp, p), x))
            for t, (tp, p) in termos.items()
        }
    return graus


def forca_das_regras(versao, graus):
    """Força de cada regra: o E fuzzy é o mínimo dos graus dos antecedentes."""
    linhas = []
    for nome, antecedentes, consequente in MODELOS[versao]["regras"]:
        forca = min(graus[v][t] for v, t in antecedentes)
        texto = " E ".join(f"{NOMES_VAR[v].split(' (')[0].lower()} é {t}" for v, t in antecedentes)
        linhas.append({"Regra": nome, "Se": texto, "Então": consequente, "Força": round(forca, 3)})
    return pd.DataFrame(linhas)


@st.cache_data(show_spinner="Calculando os scores das 15.000 vagas (só na primeira vez)...")
def carregar_e_pontuar(versao):
    df = pd.read_csv(CAMINHO_DADOS)
    df["skill_count"] = df["required_skills"].apply(conta_skills, metodo=MODELOS[versao]["contagem"])
    sim = construir_simulacao(versao)
    df["score"] = [
        pontuar(sim, {n: row[c] for n, c in COLUNAS.items()})
        for _, row in df[list(COLUNAS.values())].iterrows()
    ]
    return df


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def grafico_pertinencia(versao, var, valor=None):
    universo, termos = MODELOS[versao]["entradas"][var]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    for termo, (tp, p) in termos.items():
        ax.plot(universo, curva(universo, tp, p), lw=2, label=termo)
    if valor is not None:
        ax.axvline(valor, color="#1E2A32", ls="--", lw=1)
    ax.set_title(NOMES_VAR[var], fontsize=10)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("pertinência")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def grafico_agregacao(regras_df):
    u = SAIDA["universo"]
    agregado = np.zeros_like(u, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 3))
    for termo, (tp, p) in SAIDA["termos"].items():
        mf = curva(u, tp, p)
        nivel = regras_df.loc[regras_df["Então"] == termo, "Força"].max()
        nivel = 0.0 if pd.isna(nivel) else nivel
        cortada = np.fmin(nivel, mf)
        agregado = np.fmax(agregado, cortada)
        ax.plot(u, mf, color=CORES[termo], lw=1, ls=":")
        ax.fill_between(u, cortada, color=CORES[termo], alpha=0.45, label=f"{termo} (corte em {nivel:.2f})")
    score = fuzz.defuzz(u, agregado, "centroid") if agregado.any() else np.nan
    if not np.isnan(score):
        ax.axvline(score, color="#1E2A32", lw=2)
        ax.annotate(f"centroide = {score:.1f}", (score, 1.0), xytext=(6, -4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Atratividade (0 a 100)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, score


def histograma(serie, titulo, cor, bins=40, intervalo=(0, 100), xlabel="Score de atratividade", tamanho=(6, 3)):
    fig, ax = plt.subplots(figsize=tamanho)
    ax.hist(serie.dropna(), bins=bins, range=intervalo, color=cor, edgecolor="white")
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Vagas")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def grafico_barras_titulos(df, n=8):
    contagem = df["job_title"].value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(contagem.index, contagem.values, color=CORES["alta"])
    ax.set_title(f"Cargos mais frequentes (top {n})", fontsize=10)
    ax.set_xlabel("Vagas")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def grafico_correlacao(df):
    cols = list(COLUNAS.values())
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    nomes = [NOMES_VAR[k].split(" (")[0] for k in COLUNAS]
    ax.set_xticks(range(len(cols)), nomes, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(cols)), nomes, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlação entre as variáveis de entrada", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.title("Quão atraente é uma vaga de IA?")
st.write(
    "Um sistema de inferência fuzzy (Mamdani) que dá a cada vaga do dataset uma nota de "
    "0 a 100, do ponto de vista de quem procura emprego. A nota combina salário, "
    "experiência exigida, benefícios e quantas tecnologias a vaga pede."
)

if not CAMINHO_DADOS.exists():
    st.error(
        f"Não encontrei o dataset em `{CAMINHO_DADOS}`. Coloque o arquivo "
        "`ai_job_dataset.csv` dentro da pasta `data/`, ao lado do `app.py`, e recarregue a página."
    )
    st.stop()

aba_ideia, aba_modelo, aba_resultados, aba_simulador, aba_leitura = st.tabs(
    ["A ideia", "Como o modelo decide", "Resultados", "Simulador", "Como interpretar"]
)

# --- A ideia ---------------------------------------------------------------
with aba_ideia:
    df_c = carregar_e_pontuar("corrigido")

    col_texto, col_dados = st.columns([3, 2], gap="large")

    
    with col_texto:
        st.subheader("Por que lógica fuzzy")
        st.write(
            "Atratividade não tem fronteira nítida. Um salário de US$ 140 mil não é simplesmente "
            "\"médio\" ou \"alto\": é um pouco de cada. A lógica fuzzy permite que um valor pertença "
            "parcialmente a várias categorias e combina essas categorias com regras escritas em "
            "linguagem natural, como *se o salário é alto e os benefícios são excelentes, a vaga é "
            "muito atraente*. O resultado é uma nota explicável: dá para dizer exatamente quais "
            "regras levaram a ela."
        )
        st.subheader("O caminho de uma vaga até a nota")
        st.markdown(
            "1. **Fuzzificação.** Cada entrada vira graus de pertinência entre 0 e 1 em cada categoria.\n"
            "2. **Regras.** Cada regra recebe uma força: o *E* entre condições é o mínimo dos graus.\n"
            "3. **Agregação.** Cada conclusão (baixa, média, alta) é cortada na força da regra mais "
            "forte que aponta para ela, e as três figuras são sobrepostas.\n"
            "4. **Defuzzificação.** O centro de massa (centroide) da figura final é a nota."
        )


    # Embaixo: pequeno EDA
    st.divider()
    st.subheader("Análise exploratória (EDA)")
    st.caption("Um retrato rápido das quatro variáveis que alimentam o modelo.")

    e1, e2 = st.columns(2)
    e1.pyplot(histograma(df_c["salary_usd"], "Distribuição de salários", CORES["alta"], bins=40,
                         intervalo=(df_c["salary_usd"].min(), df_c["salary_usd"].max()),
                         xlabel="Salário (USD)", tamanho=(6, 2.8)), clear_figure=True)
    e2.pyplot(histograma(df_c["years_experience"], "Experiência exigida", CORES["media"], bins=21,
                         intervalo=(0, 20), xlabel="Anos de experiência", tamanho=(6, 2.8)), clear_figure=True)

    e3, e4 = st.columns(2)
    e3.pyplot(histograma(df_c["benefits_score"], "Nota de benefícios", CORES["baixa"], bins=25,
                         intervalo=(df_c["benefits_score"].min(), df_c["benefits_score"].max()),
                         xlabel="Nota (5 a 10)", tamanho=(6, 2.8)), clear_figure=True)
    e4.pyplot(grafico_barras_titulos(df_c), clear_figure=True)

    e5, e6 = st.columns([3, 2], gap="large")
    e5.pyplot(grafico_correlacao(df_c), clear_figure=True)
    resumo = df_c[list(COLUNAS.values())].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
    resumo.index = [NOMES_VAR[k] for k in COLUNAS]
    resumo.columns = ["Média", "Desvio", "Mín", "Mediana", "Máx"]
    e6.caption("Estatísticas descritivas")
    e6.dataframe(resumo, width="stretch")
    nulos = int(df_c[list(COLUNAS.values())].isna().sum().sum())
    e6.caption(f"Valores ausentes nas variáveis de entrada: {nulos}.")

# --- Como o modelo decide ---------------------------------------------------
with aba_modelo:
    versao_vis = "corrigido"
    st.subheader("Funções de pertinência")
    st.caption("Cada curva diz o quanto um valor pertence a uma categoria, de 0 a 1.")
    cols = st.columns(2)
    for i, var in enumerate(COLUNAS):
        cols[i % 2].pyplot(grafico_pertinencia(versao_vis, var), clear_figure=True)

    st.subheader("Base de regras")
    regras_tab = forca_das_regras(versao_vis, pertinencias(versao_vis, {k: 0 for k in COLUNAS}))
    st.dataframe(regras_tab[["Regra", "Se", "Então"]], hide_index=True, width="stretch")

# --- Resultados -------------------------------------------------------------
with aba_resultados:
    df_c = carregar_e_pontuar("corrigido")

    c1, c2, c3 = st.columns(3)
    c1.metric("Vagas sem nota", int(df_c["score"].isna().sum()))
    c2.metric("Desvio-padrão da nota", f"{df_c['score'].std():.1f}")
    c3.metric("Nota média", f"{df_c['score'].mean():.1f}")

    h1 = st.columns(1)[0]
    h1.pyplot(histograma(df_c["score"], "Frequência da pontuação de atratividade das vagas", CORES["alta"]), clear_figure=True)

    st.subheader("Quanto cada variável pesa na nota")
    st.caption("Correlação entre a nota e cada entrada. Perto de 0 significa que a variável quase não influencia.")
    corr = pd.DataFrame({
        "Variável": [NOMES_VAR[v] for v in COLUNAS],
        "Corrigido": [round(df_c["score"].corr(df_c[c]), 2) for c in COLUNAS.values()],
    })
    st.dataframe(corr, hide_index=True, width="stretch")

    st.subheader("Vagas mais atraentes (modelo corrigido)")
    top = df_c.sort_values(["score", "salary_usd", "benefits_score"], ascending=False).head(10)
    st.dataframe(
        top[["job_title", "company_location", "salary_usd", "years_experience",
             "benefits_score", "skill_count", "score"]].round({"score": 1}),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"{int((df_c['score'] >= df_c['score'].max() - 1e-6).sum())} vagas empatam na nota máxima; "
        "o desempate é por salário e depois por benefícios."
    )

# --- Simulador --------------------------------------------------------------
with aba_simulador:
    st.write("Mude os valores e veja, passo a passo, como o modelo corrigido chega à nota.")
    s1, s2 = st.columns(2)
    entradas = {
        "salary": s1.slider("Salário (USD)", 30000, 400000, 140000, step=5000),
        "experience": s1.slider("Experiência exigida (anos)", 0, 20, 9),
        "benefits": s2.slider("Nota de benefícios", 5.0, 10.0, 9.0, step=0.1),
        "skills_req": s2.slider("Tecnologias exigidas", 3, 5, 5),
    }

    graus = pertinencias("corrigido", entradas)
    st.subheader("1. Fuzzificação")
    cols = st.columns(4)
    for col, var in zip(cols, COLUNAS):
        col.markdown(f"**{NOMES_VAR[var]}**")
        for termo, g in graus[var].items():
            col.progress(g, text=f"{termo}: {g:.2f}")

    st.subheader("2. Regras que dispararam")
    regras_df = forca_das_regras("corrigido", graus)
    ativas = regras_df[regras_df["Força"] > 0].sort_values("Força", ascending=False)
    st.dataframe(ativas, hide_index=True, width="stretch")

    st.subheader("3 e 4. Agregação e defuzzificação")
    fig, score = grafico_agregacao(regras_df)
    st.pyplot(fig, clear_figure=True)
    st.metric("Nota de atratividade", f"{score:.1f}")

    forcas = ativas.groupby("Então")["Força"].max()
    lideres = forcas[forcas == forcas.max()].index.tolist()
    abertura = (
        f"A conclusão mais forte é **{lideres[0]}**"
        if len(lideres) == 1
        else "As conclusões " + " e ".join(f"**{t}**" for t in lideres) + " empatam como as mais fortes"
    )
    st.info(
        f"{abertura} (força {forcas.max():.2f}). Forças por conclusão: "
        + ", ".join(f"{t} {f:.2f}" for t, f in forcas.items())
        + ". A nota é o ponto de equilíbrio entre essas conclusões."
    )

# --- Como interpretar ---------------------------------------------------------
with aba_leitura:
    st.subheader("Lendo a nota")
    st.markdown(
        "- **Abaixo de 40:** as regras apontam principalmente para atratividade baixa, em geral salário baixo "
        "ou salário médio com muitas exigências e benefícios fracos.\n"
        "- **De 40 a 60:** conclusões em conflito; a vaga tem pontos fortes e fracos que se equilibram.\n"
        "- **Acima de 60:** predominam regras de atratividade alta, puxadas por salário alto ou salário médio "
        "com bons benefícios e poucas exigências."
    )
    st.subheader("O que os resultados mostram")
    st.write(
        "O salário continua sendo o fator dominante, o que é esperado para um índice de atratividade. "
        "Depois das correções, experiência, benefícios e quantidade de tecnologias passaram a mexer na "
        "nota. A correlação com tecnologias é negativa: com salário parecido, pedir mais skills torna "
        "a vaga menos atraente, como as regras propõem."
    )
    st.subheader("Limites do modelo")
    st.write(
        "A nota nunca passa de cerca de 84 nem fica abaixo de cerca de 13. Quando todas as regras "
        "ativas concordam, o centroide fica preso no centro do conjunto de saída. Por isso muitas "
        "vagas de salário alto empatam no teto: pelas regras, elas são igualmente atraentes. "
        "O modelo também não ajusta salário por país nem por custo de vida, e as regras refletem "
        "escolhas de quem as escreveu, não um padrão aprendido dos dados."
    )