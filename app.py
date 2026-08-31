import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
from supabase import create_client


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="CCR | Predição de internação prolongada",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_PATH = Path(__file__).with_name(
    "27_deployment_bundle.joblib"
)


# =========================================================
# FUNÇÃO NOVO CADASTRO
# =========================================================

def limpar_novo_cadastro():

    chaves = [
        chave
        for chave in list(st.session_state.keys())
        if (
            chave.startswith("pred_")
            or chave.startswith("auditoria_")
            or chave in [
                "registro_auditoria",
                "prontuario",
                "data_internacao",
                "data_cirurgia",
                "ultima_predicao",
            ]
        )
    ]

    for chave in chaves:
        st.session_state.pop(chave, None)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODELO
# =========================================================

@st.cache_resource
def carregar_bundle():
    return joblib.load(BUNDLE_PATH)


bundle = carregar_bundle()

model = bundle["model"]
predictors = bundle["predictors"]
schema = bundle.get("ui_schema", {})
threshold = bundle.get("clinical_threshold")
family = bundle.get("family", "Modelo")

sensibilidade_meta = bundle.get(
    "min_sensitivity_target"
)


if threshold is None:

    st.error(
        "O bundle não possui threshold operacional definido."
    )

    st.stop()


# =========================================================
# SUPABASE
# =========================================================

@st.cache_resource
def conectar_supabase():

    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]

    return create_client(
        url,
        key,
    )


try:

    supabase = conectar_supabase()

except Exception as exc:

    st.error(
        "Não foi possível conectar ao banco de auditoria."
    )

    st.exception(exc)
    st.stop()


# =========================================================
# NOMES AMIGÁVEIS
# =========================================================

mapa_exibicao = {

    "s": "Sim",
    "n": "Não",

    "colon_direito": "Cólon direito",
    "colon_esquerdo": "Cólon esquerdo",
    "reto_inferior": "Reto inferior",
    "reto_medio": "Reto médio",
    "retossigmoide": "Retossigmoide",
    "sincronico": "Sincrônico",

    "convencional": "Convencional",
    "laparoscopica": "Laparoscópica",
}


nomes_clinicos = {

    "sexo_int":
        "Sexo",

    "idade_anos_diag":
        "Idade ao diagnóstico",

    "f_idade_anos_int":
        "Idade",

    "f_asa":
        "Classificação ASA",

    "f_abord_cirurgica":
        "Abordagem cirúrgica",

    "f_localizacao":
        "Localização do tumor",

    "f_estagio":
        "Estágio da doença",

    "f_neoadjuvancia":
        "Terapia neoadjuvante",

    "tempo_cir_min2":
        "Tempo cirúrgico",

    "num_orgaos_envolvidos":
        "Número de órgãos envolvidos",

    "tempo_int_cir_dias":
        "Tempo entre internação e cirurgia",

    "uti":
        "Necessidade de UTI",

    "urgencia":
        "Cirurgia de urgência",
}


def formatar_categoria(valor):

    if valor is None:
        return ""

    texto = str(valor)

    if texto in mapa_exibicao:
        return mapa_exibicao[texto]

    return (
        texto
        .replace("_", " ")
        .capitalize()
    )


def nome_feature_shap(nome):

    nome = (
        str(nome)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("nom__", "")
    )

    # Procura primeiro os nomes maiores
    # para evitar correspondências parciais
    for variavel_original, nome_clinico in sorted(
        nomes_clinicos.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):

        if nome == variavel_original:
            return nome_clinico

        if nome.startswith(
            variavel_original + "_"
        ):

            categoria = nome[
                len(variavel_original) + 1:
            ]

            categoria = (
                categoria
                .replace("_", " ")
                .strip()
            )

            categoria = formatar_categoria(
                categoria
            )

            return (
                f"{nome_clinico}: {categoria}"
            )

    return (
        nome
        .replace("_", " ")
        .capitalize()
    )


# =========================================================
# CAMPOS
# =========================================================

valores = {}


def criar_campo(feature):

    meta = schema.get(
        feature,
        {},
    )

    label = meta.get(
        "label",
        nomes_clinicos.get(
            feature,
            feature,
        ),
    )


    # -----------------------------------------------------
    # VARIÁVEL CALCULADA PELAS DATAS
    # -----------------------------------------------------

    if feature == "tempo_int_cir_dias":
        return


    # -----------------------------------------------------
    # NUMÉRICA
    # -----------------------------------------------------

    if meta.get("type") == "numeric":

        min_v = meta.get("min")
        max_v = meta.get("max")

        help_txt = None

        if (
            min_v is not None
            and max_v is not None
        ):

            help_txt = (
                "Faixa observada no banco de desenvolvimento: "
                f"{int(round(min_v))} a "
                f"{int(round(max_v))}."
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
            value=None,
            step=1,
            format="%d",
            placeholder="Informe o valor",
            help=help_txt,
            key=f"pred_{feature}",
        )


    # -----------------------------------------------------
    # CATEGÓRICA
    # -----------------------------------------------------

    elif meta.get("type") == "categorical":

        options = meta.get(
            "options",
            [],
        )

        if options:

            valores[feature] = st.selectbox(
                label,
                options=options,
                index=None,
                placeholder="Selecione uma opção",
                format_func=formatar_categoria,
                key=f"pred_{feature}",
            )

        else:

            valores[feature] = st.text_input(
                label,
                value="",
                key=f"pred_{feature}",
            )


    # -----------------------------------------------------
    # OUTROS
    # -----------------------------------------------------

    else:

        valores[feature] = st.text_input(
            label,
            value="",
            key=f"pred_{feature}",
        )


# =========================================================
# AUDITORIA
# =========================================================

def calcular_auditoria(
    classificacao_prevista,
    dias_reais,
):

    if dias_reais > 7:
        desfecho_real = "Prolongada"

    else:
        desfecho_real = "Não prolongada"


    if (
        classificacao_prevista == "Alto risco"
        and
        desfecho_real == "Prolongada"
    ):

        return (
            desfecho_real,
            "VP",
            "Modelo acertou",
        )


    elif (
        classificacao_prevista == "Baixo risco"
        and
        desfecho_real == "Não prolongada"
    ):

        return (
            desfecho_real,
            "VN",
            "Modelo acertou",
        )


    elif (
        classificacao_prevista == "Alto risco"
        and
        desfecho_real == "Não prolongada"
    ):

        return (
            desfecho_real,
            "FP",
            "Modelo errou",
        )


    else:

        return (
            desfecho_real,
            "FN",
            "Modelo errou",
        )


# =========================================================
# SHAP INDIVIDUAL
# =========================================================

def calcular_shap_individual(
    novo_paciente
):

    pipeline = model

    if not hasattr(
        pipeline,
        "named_steps",
    ):
        raise ValueError(
            "O modelo salvo não possui pipeline "
            "compatível com a explicação SHAP."
        )

    if (
        "preprocess"
        not in pipeline.named_steps
    ):
        raise ValueError(
            "O pipeline não possui a etapa 'preprocess'."
        )

    if (
        "model"
        not in pipeline.named_steps
    ):
        raise ValueError(
            "O pipeline não possui a etapa 'model'."
        )


    preprocessador = (
        pipeline.named_steps[
            "preprocess"
        ]
    )

    modelo_shap = (
        pipeline.named_steps[
            "model"
        ]
    )


    # -----------------------------------------------------
    # MESMO PRÉ-PROCESSAMENTO DA PREDIÇÃO
    # -----------------------------------------------------

    X_novo = novo_paciente[
        predictors
    ].copy()

    X_proc = (
        preprocessador.transform(
            X_novo
        )
    )

    if hasattr(
        X_proc,
        "toarray",
    ):
        X_proc = X_proc.toarray()


    # -----------------------------------------------------
    # NOMES DAS FEATURES TRANSFORMADAS
    # -----------------------------------------------------

    nomes_features = (
        preprocessador
        .get_feature_names_out()
    )

    nomes_features_limpos = [
        str(nome)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("nom__", "")
        for nome in nomes_features
    ]


    X_proc_df = pd.DataFrame(
        X_proc,
        columns=nomes_features_limpos,
    )


    # -----------------------------------------------------
    # COMPATIBILIDADE XGBOOST / SHAP
    # -----------------------------------------------------

    if hasattr(
        modelo_shap,
        "enable_categorical",
    ):
        modelo_shap.enable_categorical = False

    if hasattr(
        modelo_shap,
        "cat_feature_indices",
    ):
        modelo_shap.cat_feature_indices = None

    if hasattr(
        modelo_shap,
        "_xgb_enable_categorical",
    ):
        modelo_shap._xgb_enable_categorical = False


    # -----------------------------------------------------
    # TREE EXPLAINER
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        modelo_shap
    )

    explicacao = explainer(
        X_proc_df
    )

    valores_shap = np.asarray(
        explicacao.values
    )


    # -----------------------------------------------------
    # CLASSE POSITIVA
    # -----------------------------------------------------

    if valores_shap.ndim == 3:

        valores_shap = (
            valores_shap[
                0,
                :,
                1
            ]
        )

    elif valores_shap.ndim == 2:

        valores_shap = (
            valores_shap[0]
        )

    else:

        valores_shap = (
            valores_shap
            .reshape(-1)
        )


    # -----------------------------------------------------
    # TABELA
    # -----------------------------------------------------

    tabela = pd.DataFrame(
        {
            "feature":
                nomes_features_limpos,

            "feature_clinica":
                [
                    nome_feature_shap(
                        x
                    )
                    for x
                    in nomes_features_limpos
                ],

            "valor_shap":
                valores_shap,
        }
    )


    tabela[
        "impacto_absoluto"
    ] = (
        tabela[
            "valor_shap"
        ]
        .abs()
    )


    tabela = (
        tabela
        .sort_values(
            "impacto_absoluto",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    return tabela


# =========================================================
# CABEÇALHO
# =========================================================

st.title(
    "🏥 Predição de Internação Prolongada — CCR"
)

st.subheader(
    "Ferramenta de apoio à decisão no pós-operatório"
)

st.caption(
    "Protótipo em desenvolvimento. "
    "O resultado deve ser interpretado considerando "
    "o desempenho e as limitações do modelo."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## 🧠 Sobre o modelo"
)

st.sidebar.write(
    f"**Algoritmo:** {family}"
)

st.sidebar.write(
    "**Desfecho:** Internação > 7 dias"
)

st.sidebar.write(
    f"**Threshold:** {threshold:.0%}"
)

if sensibilidade_meta is not None:

    st.sidebar.write(
        f"**Meta de sensibilidade:** "
        f"{sensibilidade_meta:.0%}"
    )

st.sidebar.divider()

st.sidebar.warning(
    "Ferramenta de apoio à decisão. "
    "Não substitui avaliação clínica."
)


# =========================================================
# ABAS
# =========================================================

(
    aba_predicao,
    aba_auditoria,
    aba_explicacao,
) = st.tabs(
    [
        "🔎 Nova predição",
        "📋 Auditoria",
        "🧠 Entenda a decisão",
    ]
)


# =========================================================
# ABA 1 — NOVA PREDIÇÃO
# =========================================================

with aba_predicao:


    # -----------------------------------------------------
    # NOVO CADASTRO
    # -----------------------------------------------------

    col_titulo, col_novo = (
        st.columns(
            [4, 1]
        )
    )


    with col_titulo:

        st.markdown(
            "### Cadastro para predição"
        )


    with col_novo:

        if st.button(
            "➕ Novo cadastro",
            use_container_width=True,
        ):

            limpar_novo_cadastro()
            st.rerun()


    # =====================================================
    # IDENTIFICAÇÃO
    # =====================================================

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🪪 Identificação do paciente"
        )

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            prontuario = st.text_input(
                "Prontuário",
                value="",
                placeholder="Digite o prontuário",
                key="prontuario",
            )


        with col2:

            data_internacao = (
                st.date_input(
                    "Data da internação",
                    value=None,
                    format="DD/MM/YYYY",
                    key="data_internacao",
                )
            )


        with col3:

            data_cirurgia = (
                st.date_input(
                    "Data da cirurgia",
                    value=None,
                    format="DD/MM/YYYY",
                    key="data_cirurgia",
                )
            )


        # -------------------------------------------------
        # TEMPO INTERNAÇÃO → CIRURGIA
        # -------------------------------------------------

        tempo_int_cir_dias = None


        if (
            data_internacao
            is not None
            and
            data_cirurgia
            is not None
        ):

            if (
                data_cirurgia
                >= data_internacao
            ):

                tempo_int_cir_dias = (
                    data_cirurgia
                    - data_internacao
                ).days

                valores[
                    "tempo_int_cir_dias"
                ] = (
                    tempo_int_cir_dias
                )

                st.info(
                    "⏱ Intervalo entre internação "
                    "e cirurgia: "
                    f"{tempo_int_cir_dias} "
                    f"{'dia' if tempo_int_cir_dias == 1 else 'dias'}"
                )

            else:

                st.error(
                    "A data da cirurgia não pode ser "
                    "anterior à data da internação."
                )


    # =====================================================
    # DADOS PARA PREDIÇÃO
    # =====================================================

    st.markdown(
        "## Dados para predição"
    )


    linha1_col1, linha1_col2 = (
        st.columns(2)
    )


    # -----------------------------------------------------
    # DADOS DO PACIENTE
    # -----------------------------------------------------

    with linha1_col1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 👤 Dados do paciente"
            )

            for feature in [
                "sexo_int",
                "f_idade_anos_int",
                "idade_anos_diag",
                "f_asa",
            ]:

                if feature in predictors:
                    criar_campo(
                        feature
                    )


    # -----------------------------------------------------
    # DADOS ONCOLÓGICOS
    # -----------------------------------------------------

    with linha1_col2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🩺 Dados oncológicos"
            )

            for feature in [
                "f_localizacao",
                "f_estagio",
                "f_neoadjuvancia",
            ]:

                if feature in predictors:
                    criar_campo(
                        feature
                    )


    linha2_col1, linha2_col2 = (
        st.columns(2)
    )


    # -----------------------------------------------------
    # PROCEDIMENTO CIRÚRGICO
    # -----------------------------------------------------

    with linha2_col1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🏥 Procedimento cirúrgico"
            )

            for feature in [
                "f_abord_cirurgica",
                "tempo_cir_min2",
                "num_orgaos_envolvidos",
            ]:

                if feature in predictors:
                    criar_campo(
                        feature
                    )


    # -----------------------------------------------------
    # INTERNAÇÃO E CUIDADOS
    # -----------------------------------------------------

    with linha2_col2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🛏️ Internação e cuidados"
            )

            for feature in [
                "uti",
                "urgencia",
            ]:

                if feature in predictors:
                    criar_campo(
                        feature
                    )


            if (
                tempo_int_cir_dias
                is not None
            ):

                st.metric(
                    "Intervalo internação → cirurgia",
                    f"{tempo_int_cir_dias} dias",
                )

            else:

                st.caption(
                    "O intervalo entre internação "
                    "e cirurgia será calculado "
                    "automaticamente."
                )


    # =====================================================
    # PREDITORES NÃO MAPEADOS
    # =====================================================

    faltantes = [
        feature
        for feature in predictors
        if (
            feature not in valores
            and
            feature != "tempo_int_cir_dias"
        )
    ]


    if faltantes:

        with st.expander(
            "Outras variáveis"
        ):

            for feature in faltantes:

                criar_campo(
                    feature
                )


    # =====================================================
    # BOTÃO CALCULAR
    # =====================================================

    st.write("")

    calcular = st.button(
        "🧠 Calcular risco de internação prolongada",
        type="primary",
        use_container_width=True,
    )


    # =====================================================
    # PREDIÇÃO
    # =====================================================

    if calcular:


        # -------------------------------------------------
        # IDENTIFICAÇÃO
        # -------------------------------------------------

        if not prontuario.strip():

            st.warning(
                "Informe o prontuário."
            )

            st.stop()


        if data_internacao is None:

            st.warning(
                "Informe a data da internação."
            )

            st.stop()


        if data_cirurgia is None:

            st.warning(
                "Informe a data da cirurgia."
            )

            st.stop()


        if (
            data_cirurgia
            < data_internacao
        ):

            st.error(
                "A data da cirurgia não pode ser "
                "anterior à data da internação."
            )

            st.stop()


        if (
            tempo_int_cir_dias
            is None
        ):

            st.error(
                "Não foi possível calcular "
                "o intervalo entre internação "
                "e cirurgia."
            )

            st.stop()


        valores[
            "tempo_int_cir_dias"
        ] = int(
            tempo_int_cir_dias
        )


        # -------------------------------------------------
        # VALIDAÇÃO DOS PREDITORES
        # -------------------------------------------------

        campos_faltantes = []


        for feature in predictors:

            valor = valores.get(
                feature
            )

            if (
                valor is None
                or
                (
                    isinstance(
                        valor,
                        str,
                    )
                    and
                    not valor.strip()
                )
            ):

                label = (
                    schema
                    .get(
                        feature,
                        {},
                    )
                    .get(
                        "label",
                        nomes_clinicos.get(
                            feature,
                            feature,
                        ),
                    )
                )

                campos_faltantes.append(
                    label
                )


        if campos_faltantes:

            st.error(
                "Preencha todos os campos "
                "antes de calcular a predição."
            )

            st.write(
                "**Campos pendentes:**"
            )

            for campo in campos_faltantes:

                st.write(
                    f"- {campo}"
                )

            st.stop()


        # -------------------------------------------------
        # DATAFRAME
        # -------------------------------------------------

        novo_paciente = (
            pd.DataFrame(
                [valores],
                columns=predictors,
            )
        )


        # -------------------------------------------------
        # PREDIÇÃO
        # -------------------------------------------------

        try:

            prob = float(
                model.predict_proba(
                    novo_paciente
                )[0, 1]
            )

        except Exception as exc:

            st.error(
                "Não foi possível calcular "
                "a predição."
            )

            st.exception(exc)
            st.stop()


        classificacao_prevista = (
            "Alto risco"
            if prob >= threshold
            else "Baixo risco"
        )


        id_predicao = str(
            uuid4()
        )


        # -------------------------------------------------
        # REGISTRO
        # -------------------------------------------------

        registro = {

            "id_predicao":
                id_predicao,

            "prontuario":
                prontuario.strip(),

            "data_internacao":
                data_internacao.isoformat(),

            "data_cirurgia":
                data_cirurgia.isoformat(),

            "probabilidade":
                prob,

            "threshold":
                float(threshold),

            "classificacao_prevista":
                classificacao_prevista,

            "modelo":
                str(family),
        }


        for feature in predictors:

            valor = valores.get(
                feature
            )

            meta = schema.get(
                feature,
                {},
            )

            if (
                meta.get("type")
                == "numeric"
            ):

                registro[
                    feature
                ] = int(valor)

            else:

                registro[
                    feature
                ] = str(valor)


        # -------------------------------------------------
        # SUPABASE
        # -------------------------------------------------

        try:

            (
                supabase
                .table(
                    "auditoria_predicoes"
                )
                .insert(
                    registro
                )
                .execute()
            )

        except Exception as exc:

            st.error(
                "A predição foi calculada, "
                "mas não foi possível registrá-la "
                "no banco de auditoria."
            )

            st.exception(exc)
            st.stop()


        # -------------------------------------------------
        # SALVA PREDIÇÃO NA SESSÃO PARA SHAP
        # -------------------------------------------------

        st.session_state[
            "ultima_predicao"
        ] = {

            "id_predicao":
                id_predicao,

            "prontuario":
                prontuario.strip(),

            "probabilidade":
                prob,

            "classificacao":
                classificacao_prevista,

            "dados":
                novo_paciente.to_dict(
                    orient="records"
                )[0],
        }


        # =================================================
        # RESULTADO
        # =================================================

        st.markdown(
            "## Resultado da predição"
        )


        st.metric(
            "Probabilidade estimada de internação > 7 dias",
            f"{prob:.1%}",
        )


        st.progress(
            min(
                max(
                    prob,
                    0.0,
                ),
                1.0,
            )
        )


        if (
            classificacao_prevista
            == "Alto risco"
        ):

            st.error(
                "🔴 ALTO RISCO — "
                f"probabilidade {prob:.1%} ≥ "
                f"threshold {threshold:.0%}."
            )

        else:

            st.success(
                "🟢 BAIXO RISCO — "
                f"probabilidade {prob:.1%} < "
                f"threshold {threshold:.0%}."
            )


        st.success(
            "Predição registrada "
            "no banco de auditoria."
        )


        st.info(
            "Abra a guia **🧠 Entenda a decisão** "
            "para visualizar os fatores que mais "
            "influenciaram esta previsão."
        )


        with st.expander(
            "Detalhes do registro"
        ):

            st.write(
                f"**Prontuário:** "
                f"{prontuario}"
            )

            st.write(
                f"**Internação:** "
                f"{data_internacao.strftime('%d/%m/%Y')}"
            )

            st.write(
                f"**Cirurgia:** "
                f"{data_cirurgia.strftime('%d/%m/%Y')}"
            )

            st.write(
                "**Intervalo internação → cirurgia:** "
                f"{tempo_int_cir_dias} dias"
            )

            st.write(
                "**ID da predição:**"
            )

            st.code(
                id_predicao
            )


# =========================================================
# ABA 2 — AUDITORIA
# =========================================================

with aba_auditoria:


    st.markdown(
        "### 1️⃣ Localizar paciente"
    )


    with st.container(
        border=True
    ):

        col_busca, col_botao = (
            st.columns(
                [3, 1]
            )
        )


        with col_busca:

            prontuario_busca = (
                st.text_input(
                    "Prontuário",
                    value="",
                    key="auditoria_prontuario",
                    placeholder="Digite o prontuário",
                )
            )


        with col_botao:

            st.write("")
            st.write("")

            buscar = st.button(
                "🔍 Buscar",
                key="buscar_predicao",
                use_container_width=True,
            )


    # =====================================================
    # BUSCA
    # =====================================================

    if buscar:

        st.session_state.pop(
            "registro_auditoria",
            None,
        )


        if not prontuario_busca.strip():

            st.warning(
                "Informe o prontuário."
            )

        else:

            try:

                resposta = (
                    supabase
                    .table(
                        "auditoria_predicoes"
                    )
                    .select("*")
                    .eq(
                        "prontuario",
                        prontuario_busca.strip(),
                    )
                    .order(
                        "data_predicao",
                        desc=True,
                    )
                    .execute()
                )


                registros = (
                    resposta.data
                    if resposta.data
                    else []
                )


                if not registros:

                    st.warning(
                        "Nenhuma predição encontrada "
                        "para este prontuário."
                    )

                else:

                    st.session_state[
                        "registro_auditoria"
                    ] = registros[0]


            except Exception as exc:

                st.error(
                    "Erro ao consultar "
                    "o banco de auditoria."
                )

                st.exception(exc)


    # =====================================================
    # REGISTRO LOCALIZADO
    # =====================================================

    registro_auditoria = (
        st.session_state.get(
            "registro_auditoria"
        )
    )


    if registro_auditoria:


        st.markdown(
            "### 2️⃣ Conferir predição"
        )


        with st.container(
            border=True
        ):

            data_cirurgia_registro = None

            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.write(
                    "**Prontuário:** "
                    f"{registro_auditoria.get('prontuario', '')}"
                )


                if (
                    registro_auditoria.get(
                        "data_internacao"
                    )
                ):

                    data_int = (
                        pd.to_datetime(
                            registro_auditoria[
                                "data_internacao"
                            ]
                        )
                    )

                    st.write(
                        "**Internação:** "
                        f"{data_int.strftime('%d/%m/%Y')}"
                    )


                if (
                    registro_auditoria.get(
                        "data_cirurgia"
                    )
                ):

                    data_cirurgia_registro = (
                        pd.to_datetime(
                            registro_auditoria[
                                "data_cirurgia"
                            ]
                        )
                        .date()
                    )

                    st.write(
                        "**Cirurgia:** "
                        f"{data_cirurgia_registro.strftime('%d/%m/%Y')}"
                    )


            with col2:

                met1, met2 = (
                    st.columns(2)
                )


                with met1:

                    st.metric(
                        "Probabilidade",
                        f"{float(registro_auditoria['probabilidade']):.1%}",
                    )


                with met2:

                    st.metric(
                        "Classificação",
                        registro_auditoria[
                            "classificacao_prevista"
                        ],
                    )


        # =================================================
        # AUDITORIA CONCLUÍDA
        # =================================================

        if (
            registro_auditoria.get(
                "desfecho_real"
            )
        ):

            st.markdown(
                "### ✅ Auditoria concluída"
            )


            with st.container(
                border=True
            ):


                if (
                    registro_auditoria.get(
                        "data_alta"
                    )
                ):

                    data_alta_registro = (
                        pd.to_datetime(
                            registro_auditoria[
                                "data_alta"
                            ]
                        )
                        .date()
                    )

                    st.write(
                        "**Data da alta:** "
                        f"{data_alta_registro.strftime('%d/%m/%Y')}"
                    )


                if (
                    registro_auditoria.get(
                        "dias_reais_internacao"
                    )
                    is not None
                ):

                    st.write(
                        "**Cirurgia → alta:** "
                        f"{registro_auditoria['dias_reais_internacao']} dias"
                    )


                col1, col2, col3 = (
                    st.columns(3)
                )


                with col1:

                    st.metric(
                        "Desfecho",
                        registro_auditoria[
                            "desfecho_real"
                        ],
                    )


                with col2:

                    st.metric(
                        "Tipo",
                        registro_auditoria[
                            "tipo_resultado"
                        ],
                    )


                with col3:

                    st.metric(
                        "Predição",
                        registro_auditoria[
                            "predicao"
                        ],
                    )


                if (
                    registro_auditoria[
                        "predicao"
                    ]
                    == "Modelo acertou"
                ):

                    st.success(
                        "✅ Modelo acertou "
                        "a classificação."
                    )

                else:

                    st.error(
                        "❌ Modelo errou "
                        "a classificação."
                    )


        # =================================================
        # REGISTRAR ALTA
        # =================================================

        else:

            st.markdown(
                "### 3️⃣ Registrar alta"
            )


            with st.container(
                border=True
            ):


                if (
                    data_cirurgia_registro
                    is None
                ):

                    st.error(
                        "Este registro não possui "
                        "data da cirurgia."
                    )

                    st.stop()


                data_alta = st.date_input(
                    "Data da alta",
                    value=None,
                    min_value=data_cirurgia_registro,
                    format="DD/MM/YYYY",
                    key="auditoria_data_alta",
                )


                if (
                    data_alta
                    is not None
                ):

                    dias_reais = (
                        data_alta
                        - data_cirurgia_registro
                    ).days


                    col1, col2 = (
                        st.columns(2)
                    )


                    with col1:

                        st.metric(
                            "Cirurgia → alta",
                            f"{dias_reais} dias",
                        )


                    with col2:

                        st.metric(
                            "Desfecho calculado",
                            (
                                "Prolongada"
                                if dias_reais > 7
                                else "Não prolongada"
                            ),
                        )


                    salvar_desfecho = (
                        st.button(
                            "✅ Registrar alta e auditar modelo",
                            type="primary",
                            use_container_width=True,
                        )
                    )


                    if salvar_desfecho:

                        (
                            desfecho_real,
                            tipo_resultado,
                            resultado_predicao,
                        ) = calcular_auditoria(
                            registro_auditoria[
                                "classificacao_prevista"
                            ],
                            int(
                                dias_reais
                            ),
                        )


                        atualizacao = {

                            "data_alta":
                                data_alta.isoformat(),

                            "dias_reais_internacao":
                                int(dias_reais),

                            "desfecho_real":
                                desfecho_real,

                            "tipo_resultado":
                                tipo_resultado,

                            "predicao":
                                resultado_predicao,

                            "data_registro_desfecho":
                                datetime.now(
                                    timezone.utc
                                ).isoformat(),
                        }


                        try:

                            (
                                supabase
                                .table(
                                    "auditoria_predicoes"
                                )
                                .update(
                                    atualizacao
                                )
                                .eq(
                                    "id_predicao",
                                    registro_auditoria[
                                        "id_predicao"
                                    ],
                                )
                                .execute()
                            )


                            st.markdown(
                                "### Resultado da auditoria"
                            )


                            col1, col2, col3 = (
                                st.columns(3)
                            )


                            with col1:

                                st.metric(
                                    "Desfecho",
                                    desfecho_real,
                                )


                            with col2:

                                st.metric(
                                    "Tipo",
                                    tipo_resultado,
                                )


                            with col3:

                                st.metric(
                                    "Predição",
                                    resultado_predicao,
                                )


                            if (
                                resultado_predicao
                                == "Modelo acertou"
                            ):

                                st.success(
                                    "✅ Modelo acertou "
                                    "a classificação deste caso."
                                )

                            else:

                                st.error(
                                    "❌ Modelo errou "
                                    "a classificação deste caso."
                                )


                            st.session_state.pop(
                                "registro_auditoria",
                                None,
                            )


                        except Exception as exc:

                            st.error(
                                "Não foi possível atualizar "
                                "o registro."
                            )

                            st.exception(exc)


                else:

                    st.caption(
                        "Informe a data da alta "
                        "para realizar a auditoria."
                    )


# =========================================================
# ABA 3 — ENTENDA A DECISÃO
# =========================================================

with aba_explicacao:

    st.markdown(
        "## 🧠 Entenda a decisão"
    )

    st.write(
        "Esta área apresenta uma explicação da "
        "predição individual e o contexto do "
        "threshold utilizado pelo modelo."
    )


    ultima_predicao = (
        st.session_state.get(
            "ultima_predicao"
        )
    )


    # =====================================================
    # SEM PREDIÇÃO
    # =====================================================

    if not ultima_predicao:

        st.info(
            "Realize uma nova predição para visualizar "
            "a explicação individual do modelo."
        )


    # =====================================================
    # COM PREDIÇÃO
    # =====================================================

    else:

        prob_explicacao = float(
            ultima_predicao[
                "probabilidade"
            ]
        )

        classificacao_explicacao = (
            ultima_predicao[
                "classificacao"
            ]
        )


        # -------------------------------------------------
        # RESUMO
        # -------------------------------------------------

        st.markdown(
            "### Resultado analisado"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Probabilidade estimada",
                f"{prob_explicacao:.1%}",
            )


        with col2:

            st.metric(
                "Classificação",
                classificacao_explicacao,
            )


        with col3:

            st.metric(
                "Threshold operacional",
                f"{threshold:.0%}",
            )


        if (
            classificacao_explicacao
            == "Alto risco"
        ):

            st.error(
                f"A probabilidade estimada foi "
                f"{prob_explicacao:.1%}, acima ou igual "
                f"ao threshold operacional de "
                f"{threshold:.0%}."
            )

        else:

            st.success(
                f"A probabilidade estimada foi "
                f"{prob_explicacao:.1%}, abaixo do "
                f"threshold operacional de "
                f"{threshold:.0%}."
            )


        st.divider()


        # =================================================
        # SHAP
        # =================================================

        st.markdown(
            "### 🔍 Como o modelo chegou a esta previsão?"
        )

        st.write(
            "O SHAP estima quanto cada variável "
            "contribuiu para deslocar a previsão "
            "deste paciente em direção a maior "
            "ou menor risco de internação prolongada."
        )


        dados_explicacao = (
            ultima_predicao[
                "dados"
            ]
        )

        paciente_explicacao = (
            pd.DataFrame(
                [dados_explicacao],
                columns=predictors,
            )
        )


        try:

            tabela_shap = (
                calcular_shap_individual(
                    paciente_explicacao
                )
            )


            # ---------------------------------------------
            # TOP FATORES
            # ---------------------------------------------

            aumentam = (
                tabela_shap[
                    tabela_shap[
                        "valor_shap"
                    ] > 0
                ]
                .sort_values(
                    "valor_shap",
                    ascending=False,
                )
                .head(5)
            )


            reduzem = (
                tabela_shap[
                    tabela_shap[
                        "valor_shap"
                    ] < 0
                ]
                .sort_values(
                    "valor_shap",
                    ascending=True,
                )
                .head(5)
            )


            col_aumentam, col_reduzem = (
                st.columns(2)
            )


            # ---------------------------------------------
            # AUMENTAM
            # ---------------------------------------------

            with col_aumentam:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### ⬆️ Fatores que aumentaram a estimativa"
                    )

                    if aumentam.empty:

                        st.caption(
                            "Nenhuma contribuição positiva "
                            "relevante foi identificada."
                        )

                    else:

                        for _, linha in aumentam.iterrows():

                            st.write(
                                "• "
                                f"{linha['feature_clinica']}"
                            )


            # ---------------------------------------------
            # REDUZEM
            # ---------------------------------------------

            with col_reduzem:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### ⬇️ Fatores que reduziram a estimativa"
                    )

                    if reduzem.empty:

                        st.caption(
                            "Nenhuma contribuição negativa "
                            "relevante foi identificada."
                        )

                    else:

                        for _, linha in reduzem.iterrows():

                            st.write(
                                "• "
                                f"{linha['feature_clinica']}"
                            )


            # =================================================
            # GRÁFICO SHAP INDIVIDUAL
            # =================================================

            st.markdown(
                "### Contribuição das variáveis"
            )


            top_plot = (
                tabela_shap
                .head(10)
                .copy()
            )


            # Ordem visual
            top_plot = (
                top_plot
                .sort_values(
                    "valor_shap",
                    ascending=True,
                )
            )


            fig, ax = plt.subplots(
                figsize=(9, 6)
            )


            cores = [
                "#C62828"
                if valor > 0
                else "#2E7D32"
                for valor
                in top_plot[
                    "valor_shap"
                ]
            ]


            ax.barh(
                top_plot[
                    "feature_clinica"
                ],
                top_plot[
                    "valor_shap"
                ],
                color=cores,
            )


            ax.axvline(
                0,
                color="black",
                linewidth=1,
            )


            ax.set_xlabel(
                "Valor SHAP"
            )

            ax.set_ylabel(
                ""
            )

            ax.set_title(
                "Principais contribuições para esta predição"
            )


            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True,
            )

            plt.close(fig)


            st.caption(
                "Barras à direita de zero aumentam "
                "a estimativa de internação prolongada; "
                "barras à esquerda reduzem a estimativa."
            )


            st.warning(
                "SHAP explica a contribuição das variáveis "
                "para esta predição específica. "
                "Não demonstra relação causal."
            )


            with st.expander(
                "Ver valores SHAP"
            ):

                tabela_exibicao = (
                    tabela_shap[
                        [
                            "feature_clinica",
                            "valor_shap",
                        ]
                    ]
                    .copy()
                )

                tabela_exibicao.columns = [
                    "Variável",
                    "Valor SHAP",
                ]

                st.dataframe(
                    tabela_exibicao,
                    use_container_width=True,
                    hide_index=True,
                )


        except Exception as exc:

            st.warning(
                "A predição foi realizada normalmente, "
                "mas não foi possível gerar a explicação "
                "SHAP deste caso."
            )

            with st.expander(
                "Detalhes técnicos do SHAP"
            ):

                st.exception(exc)


        # =================================================
        # DCA
        # =================================================

        st.divider()

        st.markdown(
            "## 📈 Utilidade clínica da decisão"
        )


        st.write(
            "A **Decision Curve Analysis (DCA)** foi utilizada "
            "para avaliar se a utilização do modelo oferece "
            "benefício líquido em comparação com duas "
            "estratégias de referência: considerar todos os "
            "pacientes como de alto risco ou considerar todos "
            "como de baixo risco."
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Threshold operacional",
                f"{threshold:.0%}",
            )


        with col2:

            st.metric(
                "Faixa contínua de benefício líquido",
                "21%–80%",
            )


        with col3:

            st.metric(
                "Meta de sensibilidade",
                (
                    f"{sensibilidade_meta:.0%}"
                    if sensibilidade_meta
                    is not None
                    else "80%"
                ),
            )


        st.info(
            "Na amostra de desenvolvimento, o modelo "
            "apresentou benefício líquido superior às "
            "estratégias de tratar todos e tratar ninguém "
            "em uma faixa contínua de thresholds entre "
            "**21% e 80%**. O threshold operacional de "
            f"**{threshold:.0%}** encontra-se dentro dessa faixa."
        )


        st.markdown(
            "### O que significa a faixa de 21% a 80%?"
        )


        st.write(
            "Essa faixa **não significa que probabilidades "
            "entre 21% e 80% sejam mais precisas**. "
            "Ela significa que, na análise de decisão realizada "
            "na amostra de desenvolvimento, utilizar o modelo "
            "para orientar decisões dentro dessa faixa de "
            "thresholds apresentou **benefício líquido potencial** "
            "em relação às estratégias de tratar todos ou "
            "tratar ninguém."
        )


        st.markdown(
            "### Por que foi escolhido 42%?"
        )


        st.write(
            "O threshold operacional não foi escolhido apenas "
            "porque estava dentro da faixa favorável da DCA. "
            "A regra de desenvolvimento exigiu simultaneamente:"
        )


        st.markdown(
            """
- atingir a meta mínima de **sensibilidade de 80%**;
- apresentar benefício líquido superior à estratégia de **tratar todos**;
- apresentar benefício líquido superior à estratégia de **tratar ninguém**;
- entre os thresholds elegíveis, selecionar o **maior threshold**, buscando manter a sensibilidade mínima e reduzir falsos positivos.
            """
        )


        st.success(
            f"Assim, o ponto de corte de **{threshold:.0%}** "
            "representa o threshold operacional definido para "
            "transformar a probabilidade estimada em uma "
            "classificação de alto ou baixo risco."
        )


        with st.expander(
            "Observação metodológica sobre a DCA"
        ):

            st.write(
                "Além da faixa contínua de 21% a 80%, "
                "a análise identificou um ponto isolado em 14% "
                "no qual o modelo também superou as duas "
                "estratégias de referência. Esse ponto isolado "
                "não foi apresentado como parte da faixa contínua."
            )

            st.write(
                "Os resultados de DCA refletem a amostra de "
                "desenvolvimento e devem ser confirmados em "
                "validação externa e/ou prospectiva antes de "
                "uso assistencial."
            )


# =========================================================
# INFORMAÇÕES METODOLÓGICAS
# =========================================================

st.divider()


with st.expander(
    "ℹ️ Informações metodológicas"
):

    st.write(
        "Internação prolongada é definida como "
        "mais de 7 dias entre a cirurgia e a alta."
    )

    st.write(
        "O intervalo entre internação e cirurgia "
        "é calculado automaticamente a partir das datas."
    )

    st.write(
        "A data da alta determina automaticamente "
        "o tempo real de internação após a cirurgia."
    )

    st.markdown(
        """
**Auditoria**

- **VP:** alto risco + internação prolongada.
- **VN:** baixo risco + internação não prolongada.
- **FP:** alto risco + internação não prolongada.
- **FN:** baixo risco + internação prolongada.

**VP e VN → Modelo acertou**  
**FP e FN → Modelo errou**
        """
    )

    st.markdown(
        """
**Explicabilidade**

O SHAP mostra a contribuição das variáveis para a predição
individual. Valores SHAP positivos deslocam a previsão em
direção à classe de internação prolongada e valores negativos
na direção oposta. A interpretação é associativa e não causal.
        """
    )
