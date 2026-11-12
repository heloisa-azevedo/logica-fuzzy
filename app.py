"""
Atividade de lógica fuzzy: índice de atratividade de vagas de IA.

Rodar com:  streamlit run app.py
Estrutura esperada:
    app.py
    .streamlit/config.toml   (tema; o Streamlit só lê esse arquivo dentro de .streamlit/)
    data/ai_job_dataset.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from skfuzzy import control as ctrl

st.set_page_config(
    page_title="Lógica Fuzzy · Atratividade de vagas de IA",
    page_icon="📊",
    layout="wide",
)

CAMINHO_DADOS = Path(__file__).parent / "data" / "ai_job_dataset.csv"

# ---------------------------------------------------------------------------
# Paleta e textos
# ---------------------------------------------------------------------------
TEXTO = "#1E2A32"
TEXTO_SUAVE = "#4A5863"
BORDA = "#DDE3E7"
FUNDO_SUAVE = "#EEF2F4"
CORES = {"baixa": "#B5523B", "media": "#A97C1B", "alta": "#2F6F73"}
CORES_TERMO = {  # cores das curvas de pertinência das entradas (baixo → alto)
    0: CORES["baixa"],
    1: CORES["media"],
    2: CORES["alta"],
}
PILULA = {  # fundo, texto
    "baixa": ("#F6E3DE", "#8E3A27"),
    "media": ("#F7EBCF", "#7A5A12"),
    "alta": ("#DCEBEC", "#1F5357"),
}
ROTULO = {
    "baixo": "baixo", "medio": "médio", "alto": "alto",
    "junior": "júnior", "pleno": "pleno", "senior": "sênior",
    "ruim": "ruim", "bom": "bom", "excelente": "excelente",
    "poucas": "poucas", "intermediarias": "intermediárias", "muitas": "muitas",
    "baixa": "baixa", "media": "média", "alta": "alta",
}
NOMES_VAR = {
    "salary": "Salário (USD)",
    "experience": "Experiência exigida (anos)",
    "benefits": "Nota de benefícios",
    "skills_req": "Tecnologias exigidas",
}
NOMES_CURTOS = {"salary": "Salário", "experience": "Experiência", "benefits": "Benefícios", "skills_req": "Tecnologias"}
COLUNAS = {
    "salary": "salary_usd",
    "experience": "years_experience",
    "benefits": "benefits_score",
    "skills_req": "skill_count",
}
NIVEIS = {"EN": "Júnior", "MI": "Pleno", "SE": "Sênior", "EX": "Executivo"}

TECHS = {
    "python", "deep learning", "pytorch", "tensorflow", "aws", "azure", "gcp",
    "docker", "kubernetes", "sql", "nlp", "computer vision", "mlops", "spark",
    "hadoop", "tableau", "data visualization", "statistics", "mathematics",
    "linux", "git", "r", "scala", "java",
}

# ---------------------------------------------------------------------------
# Definição do modelo
# Cada função de pertinência é (tipo, parâmetros): "tri" = triangular,
# "trap" = trapezoidal. Os limites seguem a distribuição real do dataset.
# ---------------------------------------------------------------------------
ENTRADAS = {
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
}
SAIDA_UNIVERSO = np.arange(0, 101, 1)
SAIDA_TERMOS = {
    "baixa": ("tri", [0, 0, 40]),
    "media": ("tri", [30, 50, 70]),
    "alta": ("trap", [60, 80, 100, 100]),
}

# Texto que explica cada categoria, mostrado ao lado do gráfico de pertinência
DESCRICAO_TERMOS = {
    "salary": {
        "baixo": "Pertinência total até US$ 55 mil, cai a zero em 100 mil.",
        "medio": "De 60 a 160 mil, com pertinência total entre 90 e 120 mil.",
        "alto": "Começa em 120 mil e é total a partir de 180 mil.",
    },
    "experience": {
        "junior": "Pertinência total até 1 ano, some em 4 anos.",
        "pleno": "De 2 a 10 anos, com pico em 6 anos.",
        "senior": "Começa em 8 anos e é total a partir de 13.",
    },
    "benefits": {
        "ruim": "Máxima em 5,0 e some em 6,5.",
        "bom": "De 5,8 a 9,2, com pico em 7,5.",
        "excelente": "Começa em 8,5 e é máxima em 10.",
    },
    "skills_req": {
        "poucas": "Máxima com 3 tecnologias, some com 4.",
        "intermediarias": "Pico com 4 tecnologias.",
        "muitas": "Começa com 4 e é máxima com 5.",
    },
}

# (nome, antecedentes, conclusão, grupo)
REGRAS = [
    ("R1", [("salary", "baixo")], "baixa", "Salário como base"),
    ("R2", [("salary", "medio")], "media", "Salário como base"),
    ("R3", [("salary", "alto")], "alta", "Salário como base"),
    ("R4", [("salary", "alto"), ("benefits", "excelente")], "alta", "Benefícios"),
    ("R5", [("salary", "medio"), ("benefits", "bom")], "media", "Benefícios"),
    ("R6", [("salary", "medio"), ("benefits", "excelente")], "alta", "Benefícios"),
    ("R7", [("salary", "medio"), ("benefits", "ruim")], "baixa", "Benefícios"),
    ("R8", [("salary", "baixo"), ("benefits", "ruim")], "baixa", "Benefícios"),
    ("R9", [("salary", "baixo"), ("benefits", "excelente")], "media", "Benefícios"),
    ("R10", [("salary", "alto"), ("benefits", "ruim")], "media", "Benefícios"),
    ("R11", [("salary", "alto"), ("experience", "junior")], "alta", "Experiência"),
    ("R12", [("salary", "alto"), ("experience", "senior")], "alta", "Experiência"),
    ("R13", [("salary", "medio"), ("experience", "pleno")], "media", "Experiência"),
    ("R14", [("salary", "baixo"), ("experience", "senior")], "baixa", "Experiência"),
    ("R15", [("salary", "baixo"), ("experience", "junior")], "media", "Experiência"),
    ("R16", [("salary", "alto"), ("skills_req", "poucas")], "alta", "Tecnologias"),
    ("R17", [("salary", "alto"), ("skills_req", "muitas")], "alta", "Tecnologias"),
    ("R18", [("salary", "medio"), ("skills_req", "poucas")], "alta", "Tecnologias"),
    ("R19", [("salary", "medio"), ("skills_req", "intermediarias")], "media", "Tecnologias"),
    ("R20", [("salary", "medio"), ("skills_req", "muitas")], "baixa", "Tecnologias"),
    ("R21", [("salary", "baixo"), ("skills_req", "poucas")], "media", "Tecnologias"),
    ("R22", [("salary", "baixo"), ("skills_req", "muitas")], "baixa", "Tecnologias"),
]

CENARIOS = {
    "Personalizado": None,
    "Salário médio, muitas exigências": (140000, 9, 9.0, 5),
    "Mesma vaga, poucas exigências": (140000, 9, 9.0, 3),
    "Salário alto": (250000, 9, 9.0, 5),
    "Salário baixo, vaga júnior": (45000, 1, 9.0, 5),
    "Salário baixo, sênior, benefícios fracos": (45000, 12, 5.5, 5),
}


# ---------------------------------------------------------------------------
# Funções do modelo
# ---------------------------------------------------------------------------
def curva(universo, tipo, params):
    return fuzz.trimf(universo, params) if tipo == "tri" else fuzz.trapmf(universo, params)


def conta_skills(texto):
    """Conta quantas das tecnologias-chave aparecem na lista de skills da vaga."""
    if pd.isna(texto):
        return 0
    itens = {s.strip().lower() for s in str(texto).split(",") if s.strip()}
    return len(itens & TECHS)


@st.cache_resource
def construir_simulacao():
    variaveis = {}
    for nome, (universo, termos) in ENTRADAS.items():
        v = ctrl.Antecedent(universo, nome)
        for termo, (tipo, p) in termos.items():
            v[termo] = curva(universo, tipo, p)
        variaveis[nome] = v
    saida = ctrl.Consequent(SAIDA_UNIVERSO, "attractiveness")
    for termo, (tipo, p) in SAIDA_TERMOS.items():
        saida[termo] = curva(SAIDA_UNIVERSO, tipo, p)

    regras = []
    for _, antecedentes, conclusao, _ in REGRAS:
        cond = variaveis[antecedentes[0][0]][antecedentes[0][1]]
        for var, termo in antecedentes[1:]:
            cond = cond & variaveis[var][termo]
        regras.append(ctrl.Rule(cond, saida[conclusao]))
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(regras))


def pontuar(simulacao, entradas):
    try:
        for nome, valor in entradas.items():
            simulacao.input[nome] = valor
        simulacao.compute()
        return simulacao.output["attractiveness"]
    except Exception:  # nenhuma regra ativada
        return np.nan


def pertinencias(entradas):
    """Grau de pertinência de cada valor de entrada em cada termo."""
    graus = {}
    for nome, (universo, termos) in ENTRADAS.items():
        x = np.clip(entradas[nome], universo.min(), universo.max())
        graus[nome] = {
            t: float(fuzz.interp_membership(universo, curva(universo, tp, p), x))
            for t, (tp, p) in termos.items()
        }
    return graus


VERBO = {"salary": "é", "experience": "é", "benefits": "são", "skills_req": "são"}
PLURAL = {"ruim": "ruins", "bom": "bons", "excelente": "excelentes"}


def descrever(antecedentes):
    return " E ".join(
        f"{NOMES_CURTOS[v].lower()} {VERBO[v]} {PLURAL.get(t, ROTULO[t]) if v == 'benefits' else ROTULO[t]}"
        for v, t in antecedentes
    )


def forca_das_regras(graus):
    """Força de cada regra: o E fuzzy é o mínimo dos graus dos antecedentes."""
    return pd.DataFrame([
        {
            "Regra": nome,
            "Se": descrever(antecedentes),
            "Então": conclusao,
            "Força": min(graus[v][t] for v, t in antecedentes),
        }
        for nome, antecedentes, conclusao, _ in REGRAS
    ])


def faixa(nota):
    if nota < 40:
        return "Pouco atraente", "baixa"
    if nota < 60:
        return "Neutra", "media"
    return "Atraente", "alta"


def br(x, casas=1):
    """Número no formato brasileiro: 1.234,5."""
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(show_spinner="Calculando a nota das 15.000 vagas (só na primeira vez)...")
def carregar_e_pontuar():
    df = pd.read_csv(CAMINHO_DADOS)
    df["skill_count"] = df["required_skills"].apply(conta_skills)
    sim = construir_simulacao()
    df["score"] = [
        pontuar(sim, {n: row[c] for n, c in COLUNAS.items()})
        for _, row in df[list(COLUNAS.values())].iterrows()
    ]
    df["faixa"] = df["score"].apply(lambda n: faixa(n)[0])
    df["nivel"] = df["experience_level"].map(NIVEIS)
    return df


# ---------------------------------------------------------------------------
# Gráficos (matplotlib com o mesmo visual em todo o app)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#C4CDD3",
    "axes.labelcolor": TEXTO_SUAVE,
    "xtick.color": TEXTO_SUAVE,
    "ytick.color": TEXTO_SUAVE,
    "text.color": TEXTO,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
})


def estilo(ax, grade_y=False):
    ax.spines[["top", "right"]].set_visible(False)
    if grade_y:
        ax.grid(axis="y", color="#EEF2F4", lw=1)
        ax.set_axisbelow(True)


def mostrar(fig, onde=st):
    onde.pyplot(fig, clear_figure=True, width="stretch")
    plt.close(fig)


def fig_pertinencia(var, marcador=None, tamanho=(6.4, 2.6), xmax=None):
    universo, termos = ENTRADAS[var]
    fig, ax = plt.subplots(figsize=tamanho)
    for i, (termo, (tp, p)) in enumerate(termos.items()):
        ax.plot(universo, curva(universo, tp, p), lw=2.4, color=CORES_TERMO[i], label=ROTULO[termo])
    if marcador is not None:
        ax.axvline(marcador, color=TEXTO, ls="--", lw=1)
        ativos = []
        for i, (termo, (tp, p)) in enumerate(termos.items()):
            g = float(fuzz.interp_membership(universo, curva(universo, tp, p), marcador))
            if g > 0:
                ax.scatter([marcador], [g], color=CORES_TERMO[i], zorder=3, s=36)
                ativos.append((g, termo))
        # rótulos empilhados para não se sobreporem
        for k, (g, termo) in enumerate(sorted(ativos, reverse=True)):
            ax.annotate(f"{ROTULO[termo]} {br(g, 2)}", (marcador, g), xytext=(8, 6 - 14 * k),
                        textcoords="offset points", fontsize=9)
    if var == "salary":
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f} mil")
    if xmax:
        ax.set_xlim(universo.min(), xmax)
    ax.set_ylim(-0.03, 1.1)
    ax.set_ylabel("pertinência")
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    estilo(ax)
    fig.tight_layout()
    return fig


def fig_saida(tamanho=(6.4, 2.6)):
    fig, ax = plt.subplots(figsize=tamanho)
    for termo, (tp, p) in SAIDA_TERMOS.items():
        ax.plot(SAIDA_UNIVERSO, curva(SAIDA_UNIVERSO, tp, p), lw=2.4, color=CORES[termo], label=ROTULO[termo])
    ax.set_ylim(-0.03, 1.1)
    ax.set_ylabel("pertinência")
    ax.set_xlabel("atratividade (0 a 100)")
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    estilo(ax)
    fig.tight_layout()
    return fig


def fig_histograma(serie, cor, bins, intervalo, xlabel, mediana_fmt=None):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.hist(serie.dropna(), bins=bins, range=intervalo, color=cor, edgecolor="white")
    if mediana_fmt:
        med = serie.median()
        ax.axvline(med, color=TEXTO, ls="--", lw=1)
        ax.annotate(f"mediana {mediana_fmt(med)}", (med, 1), xycoords=("data", "axes fraction"),
                    xytext=(5, -12), textcoords="offset points", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("vagas")
    if serie.name == "salary_usd":
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f} mil")
    estilo(ax, grade_y=True)
    fig.tight_layout()
    return fig


def fig_barras(serie, cor, formato=lambda v: br(v, 1), tamanho=None):
    altura = 0.38 * len(serie) + 0.6
    fig, ax = plt.subplots(figsize=tamanho or (6, altura))
    ax.barh(serie.index[::-1], serie.values[::-1], color=cor, height=0.62)
    for i, v in enumerate(serie.values[::-1]):
        ax.text(v, i, f" {formato(v)}", va="center", fontsize=9)
    ax.set_xlim(0, max(serie.values) * 1.14)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_correlacao(df):
    cols = list(COLUNAS.values())
    corr = df[cols].corr()
    nomes = [NOMES_CURTOS[k] for k in COLUNAS]
    cmap = LinearSegmentedColormap.from_list("petroleo", ["#F4F6F7", "#2F6F73", "#1F5357"])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.imshow(corr.abs().values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), nomes, fontsize=9)
    ax.set_yticks(range(len(cols)), nomes, fontsize=9)
    ax.tick_params(length=0)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, br(v, 2), ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else TEXTO)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def fig_agregacao(regras_df):
    u = SAIDA_UNIVERSO
    agregado = np.zeros_like(u, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for termo, (tp, p) in SAIDA_TERMOS.items():
        mf = curva(u, tp, p)
        nivel = regras_df.loc[regras_df["Então"] == termo, "Força"].max()
        nivel = 0.0 if pd.isna(nivel) else nivel
        cortada = np.fmin(nivel, mf)
        agregado = np.fmax(agregado, cortada)
        ax.plot(u, mf, color=CORES[termo], lw=1, ls=":")
        ax.fill_between(u, cortada, color=CORES[termo], alpha=0.45,
                        label=f"{ROTULO[termo]}, corte em {br(nivel, 2)}")
    score = fuzz.defuzz(u, agregado, "centroid") if agregado.any() else np.nan
    if not np.isnan(score):
        ax.axvline(score, color=TEXTO, lw=2.2)
        ax.annotate(f"centroide = {br(score)}", (score, 1.0), xytext=(7, -6),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_xlabel("atratividade (0 a 100)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    estilo(ax)
    fig.tight_layout()
    return fig, score


def fig_notas(serie):
    fig, ax = plt.subplots(figsize=(7, 3))
    for ini, fim, termo in [(0, 40, "baixa"), (40, 60, "media"), (60, 100, "alta")]:
        ax.axvspan(ini, fim, color=CORES[termo], alpha=0.08, lw=0)
    ax.hist(serie.dropna(), bins=20, range=(0, 100), color="#3E4C59", edgecolor="white")
    med = serie.median()
    ax.axvline(med, color=TEXTO, ls="--", lw=1)
    ax.annotate(f"mediana {br(med)}", (med, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -12), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("nota de atratividade")
    ax.set_ylabel("vagas")
    estilo(ax, grade_y=True)
    fig.tight_layout()
    return fig


def fig_pesos(pesos):
    fig, ax = plt.subplots(figsize=(6, 2.4))
    nomes = list(pesos.index)[::-1]
    valores = list(pesos.values)[::-1]
    cores = [CORES["alta"] if v >= 0 else CORES["baixa"] for v in valores]
    ax.barh(nomes, valores, color=cores, height=0.6)
    ax.axvline(0, color=TEXTO, lw=1)
    for i, v in enumerate(valores):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, br(v, 2), va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlim(-0.5, 1.0)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Peças de interface
# ---------------------------------------------------------------------------
CSS = f"""
<style>
.block-container {{ max-width: 1120px; padding-top: 2.6rem; padding-bottom: 4rem; }}
h1 {{ font-size: 2.4rem !important; line-height: 1.15 !important; padding-top: 0.2rem !important; }}
h2 {{ font-size: 1.6rem !important; margin-top: 1.2rem !important; }}
.eyebrow {{ font-size: 0.8rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            color: {CORES['alta']}; margin-bottom: -0.6rem; }}
.lead {{ font-size: 1.12rem; line-height: 1.55; color: #3A4853; max-width: 760px; margin-bottom: 1rem; }}
.nota {{ font-size: 0.85rem; color: {TEXTO_SUAVE}; line-height: 1.45; }}
.pilula {{ display: inline-block; border-radius: 999px; padding: 2px 12px; font-size: 0.85rem; font-weight: 600; }}
.caixa {{ border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; line-height: 1.5; }}
.caixa b {{ display: block; margin-bottom: 2px; }}
.passo-num {{ display: inline-flex; width: 28px; height: 28px; border-radius: 14px; background: {CORES['alta']};
             color: #fff; font-weight: 700; font-size: 0.9rem; align-items: center; justify-content: center; }}
.passo-titulo {{ font-weight: 700; font-size: 1.05rem; margin: 8px 0 4px; }}
.nota-grande {{ font-size: 3.4rem; font-weight: 700; line-height: 1; margin-right: 12px; }}
.escala {{ position: relative; margin: 18px 0 4px; }}
.escala-barra {{ display: flex; height: 10px; border-radius: 5px; overflow: hidden; }}
.escala-marca {{ position: absolute; top: -4px; width: 4px; height: 18px; margin-left: -2px;
                background: {TEXTO}; border-radius: 2px; }}
.escala-rotulos {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {TEXTO_SUAVE};
                  margin-top: 6px; }}
.marca {{ display: flex; align-items: center; gap: 12px; margin: 0.2rem 0 0.6rem; }}
.marca-nome {{ font-weight: 700; font-size: 1.05rem; line-height: 1.2; }}
.marca-sub {{ font-size: 0.82rem; color: {TEXTO_SUAVE}; }}
.rotulo-lateral {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
                  color: {TEXTO_SUAVE}; margin: 0.4rem 0 0.4rem; }}
.linha-lateral {{ display: flex; justify-content: space-between; font-size: 0.9rem; padding: 3px 0; }}
.ancora a {{ display: block; color: {TEXTO}; text-decoration: none; font-size: 0.92rem; padding: 4px 0; }}
.ancora a:hover {{ color: {CORES['alta']}; }}
.rodape {{ font-size: 0.8rem; color: {TEXTO_SUAVE}; line-height: 1.5; margin-top: 1.5rem; }}
</style>
"""

LOGO_SVG = """
<svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true"><rect width="40" height="40" rx="10" fill="#2F6F73"/>
<polyline points="5,30 13,12 21,30" fill="none" stroke="#fff" stroke-width="2.2" stroke-linejoin="round"/>
<polyline points="13,30 21,12 29,30" fill="none" stroke="#fff" stroke-opacity="0.7" stroke-width="2.2" stroke-linejoin="round"/>
<polyline points="21,30 29,12 36,30" fill="none" stroke="#fff" stroke-opacity="0.45" stroke-width="2.2" stroke-linejoin="round"/></svg>
"""


def html(conteudo, onde=st):
    onde.markdown(conteudo, unsafe_allow_html=True)


def cabecalho(eyebrow, titulo, lead):
    html(f'<div class="eyebrow">{eyebrow}</div>')
    st.title(titulo, anchor=False)
    html(f'<div class="lead">{lead}</div>')


def pilula(termo, texto=None):
    fundo, cor = PILULA[termo]
    return f'<span class="pilula" style="background:{fundo};color:{cor}">{texto or ROTULO[termo]}</span>'


def caixa_faixa(termo, titulo, texto):
    fundo = {"baixa": "#F9EEEA", "media": "#FBF3E1", "alta": "#E8F1F2"}[termo]
    html(f'<div class="caixa" style="background:{fundo}"><b style="color:{PILULA[termo][1]}">{titulo}</b>{texto}</div>')


def estilo_conclusao(valor):
    fundo, cor = PILULA.get(valor, ("", ""))
    return f"background-color: {fundo}; color: {cor}; font-weight: 600" if fundo else ""


def tabela_regras(tabela, com_forca=False):
    t = tabela.copy()
    estilo_tab = t.style.map(estilo_conclusao, subset=["Então"]).format({"Então": lambda v: ROTULO[v]})
    config = {
        "Regra": st.column_config.TextColumn(width="small"),
        "Se": st.column_config.TextColumn(width="large"),
        "Então": st.column_config.TextColumn(width="small"),
    }
    if com_forca:
        config["Força"] = st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f", width="medium")
    st.dataframe(estilo_tab, hide_index=True, width="stretch", column_config=config)


def sidebar_sobre():
    html('<div class="rotulo-lateral">Sobre o modelo</div>', st.sidebar)
    linhas = [("Vagas", "15.000"), ("Entradas", str(len(ENTRADAS))), ("Regras", str(len(REGRAS)))]
    html("".join(f'<div class="linha-lateral"><span>{a}</span><b>{b}</b></div>' for a, b in linhas), st.sidebar)


def sidebar_ancoras(itens):
    html('<div class="rotulo-lateral">Nesta página</div>', st.sidebar)
    html('<div class="ancora">' + "".join(f'<a href="#{a}" target="_self">{t}</a>' for a, t in itens) + "</div>",
         st.sidebar)


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------
def pagina_visao_geral():
    df = carregar_e_pontuar()
    sidebar_sobre()

    cabecalho(
        "Visão geral",
        "Quão atraente é uma vaga de IA?",
        "Um sistema de inferência fuzzy (Mamdani) que dá a cada vaga do dataset uma nota de 0 a 100, "
        "do ponto de vista de quem procura emprego. A nota combina salário, experiência exigida, "
        "benefícios e quantas tecnologias a vaga pede.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vagas analisadas", br(len(df), 0), border=True)
    c2.metric("Variáveis de entrada", len(ENTRADAS), border=True)
    c3.metric("Regras fuzzy", len(REGRAS), border=True)
    c4.metric("Escala da nota", "0 a 100", border=True)

    col_texto, col_fig = st.columns([3, 2], gap="large", vertical_alignment="center")
    with col_texto:
        st.header("Por que lógica fuzzy", anchor=False)
        st.write(
            "Atratividade não tem fronteira nítida. Um salário de US$ 140 mil não é simplesmente "
            "“médio” ou “alto”: é um pouco de cada. A lógica fuzzy permite que um valor pertença "
            "parcialmente a várias categorias e combina essas categorias com regras em linguagem "
            "natural, como *se o salário é alto e os benefícios são excelentes, a vaga é muito atraente*."
        )
        st.write("O resultado é uma nota explicável: dá para dizer exatamente quais regras levaram a ela.")
    with col_fig:
        with st.container(border=True):
            st.markdown("**US$ 140 mil em duas categorias ao mesmo tempo**")
            mostrar(fig_pertinencia("salary", marcador=140000, tamanho=(5, 2.4), xmax=250000))

    st.header("O caminho de uma vaga até a nota", anchor=False)
    passos = [
        ("Fuzzificação", "Cada entrada vira graus de pertinência entre 0 e 1 em cada categoria."),
        ("Regras", "Cada regra recebe uma força. O <em>E</em> entre condições é o mínimo dos graus."),
        ("Agregação", "Cada conclusão é cortada na força da regra mais forte e as figuras se sobrepõem."),
        ("Defuzzificação", "O centro de massa (centroide) da figura final vira a nota."),
    ]
    for col, (i, (titulo, texto)) in zip(st.columns(4), enumerate(passos, start=1)):
        with col.container(border=True, height="stretch"):
            html(f'<span class="passo-num">{i}</span><div class="passo-titulo">{titulo}</div>{texto}')
    st.page_link(PAGINAS["simulador"], label="Ver as quatro etapas no simulador", icon=":material/arrow_forward:")

    st.divider()
    st.header("Os dados", anchor=False)
    html('<div class="lead">Um retrato rápido das quatro variáveis que alimentam o modelo.</div>')

    e1, e2 = st.columns(2)
    with e1.container(border=True):
        st.markdown("**Salário (USD)**")
        mostrar(fig_histograma(df["salary_usd"], CORES["alta"], 30,
                               (df["salary_usd"].min(), df["salary_usd"].max()), "salário",
                               mediana_fmt=lambda v: f"{br(v / 1000)} mil"))
    with e2.container(border=True):
        st.markdown("**Experiência exigida (anos)**")
        mostrar(fig_histograma(df["years_experience"], CORES["alta"], 20, (0, 20), "anos",
                               mediana_fmt=lambda v: br(v, 0)))
    e3, e4 = st.columns(2)
    with e3.container(border=True):
        st.markdown("**Nota de benefícios**")
        mostrar(fig_histograma(df["benefits_score"], CORES["alta"], 10, (5, 10), "nota",
                               mediana_fmt=lambda v: br(v)))
    with e4.container(border=True):
        st.markdown("**Cargos mais frequentes**")
        mostrar(fig_barras(df["job_title"].value_counts().head(8), CORES["alta"],
                           formato=lambda v: br(v, 0), tamanho=(6, 2.5)))

    e5, e6 = st.columns(2)
    with e5.container(border=True, height="stretch"):
        st.markdown("**Correlação entre as entradas**")
        mostrar(fig_correlacao(df))
        html('<div class="nota">Só salário e experiência andam juntos; as outras entradas são independentes.</div>')
    with e6.container(border=True, height="stretch"):
        st.markdown("**Estatísticas descritivas**")
        resumo = df[list(COLUNAS.values())].describe().T[["mean", "std", "min", "50%", "max"]]
        resumo.index = [NOMES_CURTOS[k] for k in COLUNAS]
        resumo.columns = ["Média", "Desvio", "Mín", "Mediana", "Máx"]
        st.dataframe(resumo.style.format(lambda v: br(v, 0 if abs(v) >= 1000 else 2)), width="stretch")
        nulos = int(df[list(COLUNAS.values())].isna().sum().sum())
        html(f'<div class="nota">Valores ausentes nas variáveis de entrada: {nulos}.</div>')


def pagina_modelo():
    sidebar_ancoras([("pertinencia", "Funções de pertinência"), ("regras", "Base de regras"),
                     ("nota", "Das regras à nota")])

    cabecalho(
        "O modelo fuzzy",
        "Como o modelo decide",
        f"Quatro entradas viram graus de pertinência, {len(REGRAS)} regras combinam esses graus "
        "e o centroide transforma tudo em uma nota.",
    )

    st.header("Funções de pertinência", anchor="pertinencia")
    st.write("Cada curva diz o quanto um valor pertence a uma categoria, de 0 a 1. "
             "Onde duas curvas se cruzam, o valor pertence às duas.")
    abas = st.tabs([NOMES_CURTOS[v] for v in ENTRADAS] + ["Atratividade (saída)"])
    for aba, var in zip(abas, ENTRADAS):
        with aba:
            g, t = st.columns([3, 2], gap="large")
            with g.container(border=True):
                mostrar(fig_pertinencia(var))
            with t:
                for i, (termo, desc) in enumerate(DESCRICAO_TERMOS[var].items()):
                    cor = list(PILULA.values())[i][1]
                    html(f'<div class="caixa" style="background:{FUNDO_SUAVE}">'
                         f'<b style="color:{cor}">{ROTULO[termo].capitalize()}</b>{desc}</div>')
                html('<div class="nota">Limites escolhidos pela distribuição real do dataset.</div>')
    with abas[-1]:
        g, t = st.columns([3, 2], gap="large")
        with g.container(border=True):
            mostrar(fig_saida())
        with t:
            st.write("A saída também é fuzzy: atratividade baixa, média ou alta. As regras dizem qual "
                     "dessas categorias ativar e com que força, e só no fim tudo vira um número.")

    st.header("Base de regras", anchor="regras")
    st.write("O salário é o eixo de todas as regras. As três regras-base usam só o salário e garantem que "
             "toda vaga receba uma nota; as demais ajustam essa base conforme benefícios, experiência e "
             "tecnologias.")
    tabela = pd.DataFrame([{"Grupo": g, "Regra": n, "Se": descrever(a), "Então": c} for n, a, c, g in REGRAS])
    grupos = list(dict.fromkeys(tabela["Grupo"]))
    abas = st.tabs([f"{g} · {(tabela['Grupo'] == g).sum()}" for g in grupos])
    for aba, grupo in zip(abas, grupos):
        with aba:
            tabela_regras(tabela[tabela["Grupo"] == grupo].drop(columns="Grupo"))

    st.header("Das regras à nota", anchor="nota")
    cartoes = [
        ("E = mínimo", "A força de uma regra é o menor grau entre suas condições. Salário 0,50 médio e "
                       "tecnologias 1,00 “muitas” dão força 0,50."),
        ("Agregação = máximo", "Para cada conclusão vale a regra mais forte que aponta para ela. É a altura "
                               "em que a curva de saída é cortada."),
        ("Centroide", "O centro de massa da figura formada pelas curvas cortadas é a nota final, de 0 a 100."),
    ]
    for col, (titulo, texto) in zip(st.columns(3), cartoes):
        with col.container(border=True, height="stretch"):
            html(f'<div class="eyebrow" style="margin-bottom:6px">{titulo}</div>{texto}')
    st.page_link(PAGINAS["simulador"], label="Testar com uma vaga no simulador", icon=":material/arrow_forward:")


def aplicar_cenario():
    valores = CENARIOS[st.session_state["cenario"]]
    if valores:
        for chave, v in zip(["sim_salario", "sim_exp", "sim_beneficios", "sim_tecn"], valores):
            st.session_state[chave] = v


def pagina_simulador():
    padrao = CENARIOS["Salário médio, muitas exigências"]
    for chave, v in zip(["sim_salario", "sim_exp", "sim_beneficios", "sim_tecn"], padrao):
        st.session_state.setdefault(chave, v)

    with st.sidebar:
        html('<div class="rotulo-lateral">Parâmetros da vaga</div>')
        entradas = {
            "salary": st.slider("Salário (USD)", 30000, 400000, step=5000, key="sim_salario", format="%d"),
            "experience": st.slider("Experiência (anos)", 0, 20, key="sim_exp"),
            "benefits": st.slider("Nota de benefícios", 5.0, 10.0, step=0.1, key="sim_beneficios"),
            "skills_req": st.slider("Tecnologias exigidas", 3, 5, key="sim_tecn"),
        }
        st.selectbox("Cenário pronto", list(CENARIOS), index=0, key="cenario", on_change=aplicar_cenario)

    cabecalho("Simulador", "Do perfil da vaga à nota",
              "Ajuste os parâmetros na barra lateral e acompanhe as quatro etapas do raciocínio fuzzy.")

    graus = pertinencias(entradas)
    regras_df = forca_das_regras(graus)
    fig_agg, score = fig_agregacao(regras_df)
    ativas = regras_df[regras_df["Força"] > 0].sort_values("Força", ascending=False)
    forcas = ativas.groupby("Então")["Força"].max()
    nome_faixa, termo_faixa = faixa(score)

    c1, c2 = st.columns(2, gap="medium")
    with c1.container(border=True, height="stretch"):
        html(f'<div class="nota" style="margin-bottom:10px">Nota de atratividade</div>'
             f'<span class="nota-grande">{br(score)}</span>{pilula(termo_faixa, nome_faixa)}'
             f'<div class="escala"><div class="escala-barra">'
             f'<span style="width:40%;background:#E9C5BB"></span><span style="width:20%;background:#EED9A6"></span>'
             f'<span style="width:40%;background:#B9D5D7"></span></div>'
             f'<span class="escala-marca" style="left:{score:.1f}%"></span>'
             f'<div class="escala-rotulos"><span>0</span><span>40</span><span>60</span><span>100</span></div></div>')
    with c2.container(border=True, height="stretch"):
        lideres = forcas[forcas == forcas.max()].index.tolist()
        if len(lideres) == 1:
            abertura = f"A conclusão mais forte é <b>{ROTULO[lideres[0]]}</b>"
        else:
            abertura = "As conclusões " + " e ".join(f"<b>{ROTULO[t]}</b>" for t in lideres) + \
                       " empatam como as mais fortes"
        outras = [f"{ROTULO[t]} {br(f, 2)}" for t, f in forcas.items() if t not in lideres]
        complemento = f"; {', '.join(outras)}" if outras else ""
        html(f'<div class="eyebrow" style="margin-bottom:8px">Por que essa nota</div>'
             f'{abertura} ({br(forcas.max(), 2)}){complemento}. A nota {br(score)} é o ponto de equilíbrio '
             f'entre essas conclusões, por isso a vaga fica na faixa <b>{nome_faixa.lower()}</b>.')

    st.subheader("1. Fuzzificação", anchor=False)
    for col, var in zip(st.columns(4), ENTRADAS):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{NOMES_CURTOS[var]}**")
            for termo, g in graus[var].items():
                st.progress(g, text=f"{ROTULO[termo]}: {br(g, 2)}")

    st.subheader(f"2. Regras que dispararam · {len(ativas)} de {len(REGRAS)}", anchor=False)
    tabela_regras(ativas, com_forca=True)

    st.subheader("3 e 4. Agregação e defuzzificação", anchor=False)
    with st.container(border=True):
        mostrar(fig_agg)


def pagina_resultados():
    df = carregar_e_pontuar()
    sidebar_ancoras([("distribuicao", "Distribuição das notas"), ("onde", "Onde estão as melhores vagas"),
                     ("peso", "O que pesa na nota"), ("top", "Vagas mais atraentes"),
                     ("limites", "Limites do modelo")])

    cabecalho("Resultados", f"O que o modelo diz das {br(len(df), 0)} vagas",
              "Os números do modelo aplicado ao dataset inteiro e como ler cada um deles.")

    proporcao = df["faixa"].value_counts(normalize=True)
    p_baixa, p_media, p_alta = (proporcao.get(k, 0) for k in ["Pouco atraente", "Neutra", "Atraente"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nota média", br(df["score"].mean()), border=True)
    c2.metric("Desvio-padrão", br(df["score"].std()), border=True)
    c3.metric("Vagas sem nota", int(df["score"].isna().sum()), border=True)
    with c4.container(border=True, height="stretch"):
        html(f'<div class="nota" style="margin-bottom:10px">Vagas por faixa</div>'
             f'<div class="escala-barra" style="height:12px">'
             f'<span style="width:{p_baixa:.0%};background:{CORES["baixa"]}"></span>'
             f'<span style="width:{p_media:.0%};background:#D9A441"></span>'
             f'<span style="width:{p_alta:.0%};background:{CORES["alta"]}"></span></div>'
             f'<div class="escala-rotulos" style="font-size:0.85rem">'
             f'<span style="color:{PILULA["baixa"][1]}">{p_baixa:.0%}</span>'
             f'<span style="color:{PILULA["media"][1]}">{p_media:.0%}</span>'
             f'<span style="color:{PILULA["alta"][1]}">{p_alta:.0%}</span></div>')

    st.header("Distribuição das notas", anchor="distribuicao")
    g, t = st.columns([3, 2], gap="large")
    with g.container(border=True):
        mostrar(fig_notas(df["score"]))
        html('<div class="nota">Os picos perto de 31 e de 84 são vagas em que todas as regras ativas '
             'concordam: o centroide fica no centro do conjunto de saída.</div>')
    with t:
        st.markdown("**Como ler a nota**")
        caixa_faixa("baixa", "Abaixo de 40 · pouco atraente",
                    "Predominam regras de atratividade baixa: salário baixo, ou médio com muitas exigências "
                    "e benefícios fracos.")
        caixa_faixa("media", "De 40 a 60 · neutra",
                    "Conclusões em conflito: pontos fortes e fracos se equilibram.")
        caixa_faixa("alta", "Acima de 60 · atraente",
                    "Predominam regras de atratividade alta: salário alto, ou médio com bons benefícios "
                    "e poucas exigências.")

    st.header("Onde estão as melhores vagas", anchor="onde")
    por_nivel = df.groupby("nivel")["score"].mean().reindex(NIVEIS.values())
    por_pais = df.groupby("company_location")["score"].mean().sort_values(ascending=False)
    b1, b2 = st.columns(2)
    with b1.container(border=True, height="stretch"):
        st.markdown("**Nota média por nível da vaga**")
        mostrar(fig_barras(por_nivel, CORES["alta"], tamanho=(6, 2.4)))
        html('<div class="nota">Vagas mais seniores pagam mais, e o salário é o eixo das regras.</div>')
    with b2.container(border=True, height="stretch"):
        st.markdown(f"**Nota média por país (8 maiores de {len(por_pais)})**")
        mostrar(fig_barras(por_pais.head(8), "#3E4C59", tamanho=(6, 3.2)))
        with st.expander("Ver todos os países"):
            st.dataframe(por_pais.rename("Nota média").to_frame().style.format(lambda v: br(v)),
                         width="stretch")

    st.header("O que pesa na nota", anchor="peso")
    pesos = pd.Series({NOMES_CURTOS[v]: df["score"].corr(df[c]) for v, c in COLUNAS.items()})
    g, t = st.columns(2, gap="large")
    with g.container(border=True):
        st.markdown("**Correlação de cada entrada com a nota**")
        mostrar(fig_pesos(pesos))
    with t:
        st.markdown("**O que os resultados mostram**")
        nivel_txt = ", ".join(f"{n.lower()} {br(v, 0)}" for n, v in por_nivel.items())
        st.write(
            f"O salário é o fator dominante (correlação de {br(pesos['Salário'], 2)}), como se espera de "
            f"um índice de atratividade. A nota sobe com o nível da vaga ({nivel_txt}), e o topo dos "
            "países é ocupado por mercados de salários altos."
        )
        st.write(
            f"A correlação com tecnologias é negativa ({br(pesos['Tecnologias'], 2)}): com salário parecido, "
            "pedir mais skills torna a vaga menos atraente, como as regras propõem."
        )

    st.header("Vagas mais atraentes", anchor="top")
    top = df.sort_values(["score", "salary_usd", "benefits_score"], ascending=False).head(10).round(
        {"score": 1, "benefits_score": 1})
    st.dataframe(
        top[["job_title", "company_location", "nivel", "salary_usd", "years_experience",
             "benefits_score", "skill_count", "score"]],
        hide_index=True, width="stretch",
        column_config={
            "job_title": "Cargo",
            "company_location": "País",
            "nivel": "Nível",
            "salary_usd": st.column_config.NumberColumn("Salário (USD)", format="localized"),
            "years_experience": "Exp. (anos)",
            "benefits_score": st.column_config.NumberColumn("Benefícios", format="localized"),
            "skill_count": "Tecnologias",
            "score": st.column_config.NumberColumn("Nota", format="localized"),
        },
    )
    empates = int((df["score"] >= df["score"].max() - 1e-6).sum())
    html(f'<div class="nota">{br(empates, 0)} vagas empatam na nota máxima; o desempate é por salário '
         'e depois por benefícios.</div>')

    st.header("Limites do modelo", anchor="limites")
    with st.container(border=True):
        st.write(
            "A nota fica entre cerca de 13 e 84. Quando todas as regras ativas concordam, o centroide fica "
            "preso no centro do conjunto de saída, por isso muitas vagas de salário alto empatam no teto. "
            "O modelo também não ajusta salário por custo de vida, e as regras refletem o julgamento de quem "
            "as escreveu, não um padrão aprendido dos dados. A vantagem é que esse julgamento fica explícito "
            "e pode ser discutido e alterado."
        )


# ---------------------------------------------------------------------------
# Navegação
# ---------------------------------------------------------------------------
PAGINAS = {
    "visao": st.Page(pagina_visao_geral, title="Visão geral", icon=":material/home:", default=True),
    "modelo": st.Page(pagina_modelo, title="O modelo fuzzy", icon=":material/stacked_line_chart:",
                      url_path="modelo"),
    "simulador": st.Page(pagina_simulador, title="Simulador", icon=":material/tune:", url_path="simulador"),
    "resultados": st.Page(pagina_resultados, title="Resultados", icon=":material/bar_chart:",
                          url_path="resultados"),
}

html(CSS)
pagina = st.navigation(list(PAGINAS.values()), position="hidden")

with st.sidebar:
    html(f'<div class="marca">{LOGO_SVG}<div><div class="marca-nome">Lógica Fuzzy</div>'
         f'<div class="marca-sub">Atratividade de vagas de IA</div></div></div>')
    for p in PAGINAS.values():
        st.page_link(p, width="stretch")
    st.divider()

if not CAMINHO_DADOS.exists():
    st.error(
        f"Não encontrei o dataset em `{CAMINHO_DADOS}`. Coloque o arquivo `ai_job_dataset.csv` "
        "dentro da pasta `data/`, ao lado do `app.py`, e recarregue a página."
    )
    st.stop()

pagina.run()

html('<div class="rodape">Atividade de Lógica Fuzzy<br>Catharina Gato, Felipe Nunes e Heloisa Azevedo</div>', st.sidebar)
