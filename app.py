import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="CCR | Risco de internação prolongada",
    page_icon="🏥",
    layout="wide",
)

BUNDLE_PATH = Path(__file__).with_name("27_deployment_bundle.joblib")


@st.cache_resource
def carregar_bundle():
    return joblib.load(BUNDLE_PATH)


# =========================================================
# CARREGAMENTO DO MODELO
# =========================================================

bundle = carregar_bundle()

model = bundle["model"]
predictors = bundle["predictors"]
schema = bundle.get("ui_schema", {})
threshold = bundle.get("clinical_threshold")
family = bundle.get("family", "Modelo")
sensibilidade_meta = bundle.get("min_sensitivity_target")


# =========================================================
# CABEÇALHO
# =========================================================

st.title("Predição de internação prolongada")

st.caption(
    "Pacientes submetidos à cirurgia por câncer colorretal"
)

st.info(
    "Protótipo de apoio à decisão para estimar risco de internação > 7 dias. "
    "Não substitui avaliação clínica e requer validação externa/prospectiva "
    "antes de uso assistencial."
)


# =========================================================
# VERIFICAÇÃO DO THRESHOLD
# =========================================================

if threshold is None:
    st.error(
        "O bundle não possui threshold operacional definido. "
        "Retorne ao notebook e revise a etapa de seleção do threshold."
    )
    st.stop()


# =========================================================
# BARRA LATERAL
# =========================================================

st.sidebar.header("Modelo")

st.sidebar.write(
    f"**Algoritmo:** {family}"
)

st.sidebar.write(
    f"**Threshold operacional:** {threshold:.0%}"
)

if sensibilidade_meta is not None:
    st.sidebar.write(
        f"**Meta de sensibilidade:** {sensibilidade_meta:.0%}"
    )

st.sidebar.caption(
    bundle.get("threshold_rule", "")
)


# =========================================================
# NOMES AMIGÁVEIS DAS CATEGORIAS
# =========================================================

mapa_exibicao = {

    # Sim / Não
    "s": "Sim",
    "n": "Não",

    # Localização do tumor
    "colon_direito": "Cólon direito",
    "colon_esquerdo": "Cólon esquerdo",
    "reto_inferior": "Reto inferior",
    "reto_medio": "Reto médio",
    "retossigmoide": "Retossigmoide",
    "sincronico": "Sincrônico",

    # Abordagem cirúrgica
    "convencional": "Convencional",
    "laparoscopica": "Laparoscópica",
}


def formatar_categoria(valor):

    texto = str(valor)

    if texto in mapa_exibicao:
        return mapa_exibicao[texto]

    return texto.replace("_", " ").capitalize()


# =========================================================
# FUNÇÃO PARA CRIAR CADA CAMPO
# =========================================================

valores = {}


def criar_campo(feature):

    meta = schema.get(feature, {})
    label = meta.get("label", feature)

    # -----------------------------------------------------
    # VARIÁVEIS QUANTITATIVAS DISCRETAS
    # -----------------------------------------------------

    if meta.get("type") == "numeric":

        min_v = meta.get("min")
        max_v = meta.get("max")
        median_v = meta.get("median")

        help_txt = None

        if min_v is not None and max_v is not None:
            help_txt = (
                "Faixa observada no banco de desenvolvimento: "
                f"{int(round(min_v))} a {int(round(max_v))}."
            )

        default = (
            0
            if median_v is None
            else int(round(median_v))
        )

        valores[feature] = st.number_input(
            label,
            min_value=(
                int(round(min_v))
                if min_v is not None
                else None
            ),
            max_value=(
                int(round(max_v))
                if max_v is not None
                else None
            ),
            value=default,
            step=1,
            format="%d",
            help=help_txt,
            key=feature,
        )

    # -----------------------------------------------------
    # VARIÁVEIS CATEGÓRICAS
    # -----------------------------------------------------

    elif meta.get("type") == "categorical":

        options = meta.get("options", [])

        if not options:

            valores[feature] = st.text_input(
                label,
                key=feature,
            )

        else:

            valores[feature] = st.selectbox(
                label,
                options=options,
                format_func=formatar_categoria,
                key=feature,
            )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    else:

        valores[feature] = st.text_input(
            label,
            key=feature,
        )


# =========================================================
# DADOS DO PACIENTE
# =========================================================

st.subheader("Dados do paciente")


# ---------------------------------------------------------
# 1. DADOS DEMOGRÁFICOS
# ---------------------------------------------------------

with st.container(border=True):

    st.markdown("### 👤 Dados demográficos")

    col1, col2 = st.columns(2)

    with col1:

        if "sexo_int" in predictors:
            criar_campo("sexo_int")

        if "idade_anos_diag" in predictors:
            criar_campo("idade_anos_diag")

    with col2:

        if "f_idade_anos_int" in predictors:
            criar_campo("f_idade_anos_int")


# ---------------------------------------------------------
# 2. CONDIÇÃO CLÍNICA E ONCOLÓGICA
# ---------------------------------------------------------

with st.container(border=True):

    st.markdown("### 🩺 Condição clínica e oncológica")

    col1, col2 = st.columns(2)

    with col1:

        if "f_asa" in predictors:
            criar_campo("f_asa")

        if "f_estagio" in predictors:
            criar_campo("f_estagio")

    with col2:

        if "f_localizacao" in predictors:
            criar_campo("f_localizacao")

        if "f_neoadjuvancia" in predictors:
            criar_campo("f_neoadjuvancia")


# ---------------------------------------------------------
# 3. PROCEDIMENTO CIRÚRGICO
# ---------------------------------------------------------

with st.container(border=True):

    st.markdown("### 🏥 Procedimento cirúrgico")

    col1, col2 = st.columns(2)

    with col1:

        if "f_abord_cirurgica" in predictors:
            criar_campo("f_abord_cirurgica")

        if "tempo_cir_min2" in predictors:
            criar_campo("tempo_cir_min2")

    with col2:

        if "num_orgaos_envolvidos" in predictors:
            criar_campo("num_orgaos_envolvidos")

        if "urgencia" in predictors:
            criar_campo("urgencia")


# ---------------------------------------------------------
# 4. INTERNAÇÃO E CUIDADOS INTENSIVOS
# ---------------------------------------------------------

with st.container(border=True):

    st.markdown("### 🛏️ Internação e cuidados intensivos")

    col1, col2 = st.columns(2)

    with col1:

        if "tempo_int_cir_dias" in predictors:
            criar_campo("tempo_int_cir_dias")

    with col2:

        if "uti" in predictors:
            criar_campo("uti")


# =========================================================
# SEGURANÇA: PREDITORES NÃO ORGANIZADOS
# =========================================================

faltantes = [
    feature
    for feature in predictors
    if feature not in valores
]

if faltantes:

    st.warning(
        "Existem variáveis do modelo ainda não organizadas "
        "nos grupos principais."
    )

    with st.expander("Outras variáveis"):

        for feature in faltantes:
            criar_campo(feature)


# =========================================================
# BOTÃO
# =========================================================

st.divider()

st.markdown("## Predição")

calcular = st.button(
    "Calcular risco de internação prolongada",
    type="primary",
    use_container_width=True,
)


# =========================================================
# PREDIÇÃO
# =========================================================

if calcular:

    novo_paciente = pd.DataFrame(
        [valores],
        columns=predictors,
    )

    try:

        prob = float(
            model.predict_proba(
                novo_paciente
            )[0, 1]
        )

    except Exception as exc:

        st.error(
            "Não foi possível calcular a predição "
            "com os dados informados."
        )

        st.exception(exc)
        st.stop()


    alto_risco = prob >= threshold


    # =====================================================
    # PAINEL PRINCIPAL DO RESULTADO
    # =====================================================

    st.markdown("## Resultado")

    with st.container(border=True):

        # -------------------------------------------------
        # LINHA PRINCIPAL
        # -------------------------------------------------

        col_prob, col_risco = st.columns(
            [1.3, 1]
        )

        with col_prob:

            st.metric(
                "Probabilidade de internação > 7 dias",
                f"{prob:.1%}",
            )

        with col_risco:

            if alto_risco:

                st.metric(
                    "Classificação",
                    "ALTO RISCO",
                )

            else:

                st.metric(
                    "Classificação",
                    "BAIXO RISCO",
                )


        # -------------------------------------------------
        # BARRA DE PROBABILIDADE
        # -------------------------------------------------

        st.markdown("**Probabilidade estimada**")

        st.progress(
            min(
                max(prob, 0.0),
                1.0,
            )
        )


        # -------------------------------------------------
        # INTERPRETAÇÃO
        # -------------------------------------------------

        if alto_risco:

            st.error(
                f"⚠️ **Alto risco de internação prolongada.** "
                f"A probabilidade estimada foi de **{prob:.1%}**, "
                f"igual ou superior ao threshold operacional "
                f"de **{threshold:.0%}**."
            )

        else:

            st.success(
                f"✅ **Baixo risco de internação prolongada.** "
                f"A probabilidade estimada foi de **{prob:.1%}**, "
                f"inferior ao threshold operacional "
                f"de **{threshold:.0%}**."
            )


        # -------------------------------------------------
        # INFORMAÇÕES OPERACIONAIS
        # -------------------------------------------------

        st.markdown("#### Parâmetros de decisão")

        if sensibilidade_meta is not None:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Threshold operacional",
                    f"{threshold:.0%}",
                )

            with col2:

                st.metric(
                    "Meta de sensibilidade",
                    f"{sensibilidade_meta:.0%}",
                )

        else:

            st.metric(
                "Threshold operacional",
                f"{threshold:.0%}",
            )


        st.caption(
            "A classificação resulta da comparação entre a "
            "probabilidade estimada pelo modelo e o threshold "
            "operacional definido no notebook."
        )


    # =====================================================
    # AVISO DE INTERPRETAÇÃO
    # =====================================================

    st.info(
        "A probabilidade deve ser interpretada em conjunto "
        "com o desempenho, a calibração e as limitações do modelo. "
        "O resultado não substitui a avaliação clínica."
    )


    # =====================================================
    # DADOS UTILIZADOS
    # =====================================================

    with st.expander(
        "Dados usados na predição"
    ):

        exibicao = (
            novo_paciente
            .T
            .reset_index()
        )

        exibicao.columns = [
            "Variável",
            "Valor",
        ]

        st.dataframe(
            exibicao,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# SOBRE O PROTÓTIPO
# =========================================================

st.divider()

with st.expander(
    "Sobre este protótipo"
):

    st.write(
        "O modelo foi desenvolvido para o desfecho de "
        "internação prolongada (>7 dias) após cirurgia por "
        "câncer colorretal."
    )

    st.write(
        "A aplicação Streamlit utiliza o pipeline previamente "
        "treinado e salvo. A aplicação não realiza novo "
        "treinamento, otimização de hiperparâmetros ou "
        "recalibração durante a predição."
    )

    st.write(
        "O resultado deve ser considerado uma ferramenta "
        "experimental de apoio à decisão e requer validação "
        "externa/prospectiva antes de eventual utilização "
        "assistencial."
    )
