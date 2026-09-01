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
# CONFIGURAÇÃO
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
# NOVO CADASTRO
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

    return joblib.load(
        BUNDLE_PATH
    )


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
# CONSULTAR BASE COMPLETA
# =========================================================

def carregar_todos_registros():

    todos = []

    tamanho_lote = 1000
    inicio = 0

    while True:

        fim = (
            inicio
            + tamanho_lote
            - 1
        )

        resposta = (
            supabase
            .table(
                "auditoria_predicoes"
            )
            .select("*")
            .order(
                "data_predicao",
                desc=True,
            )
            .range(
                inicio,
                fim,
            )
            .execute()
        )

        lote = (
            resposta.data
            if resposta.data
            else []
        )

        todos.extend(
            lote
        )

        if len(lote) < tamanho_lote:
            break

        inicio += tamanho_lote

    return todos


# =========================================================
# NOMES E CATEGORIAS
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

    "f_idade_anos_int":
        "Idade",

    "idade_anos_diag":
        "Idade ao diagnóstico",

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


def formatar_valor_paciente(
    feature,
    valor,
):

    if valor is None:
        return "Não informado"

    meta = schema.get(
        feature,
        {},
    )

    if (
        meta.get("type")
        == "categorical"
    ):

        return formatar_categoria(
            valor
        )

    if isinstance(
        valor,
        float,
    ):

        if valor.is_integer():

            return str(
                int(valor)
            )

    return str(valor)


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


    if feature == "tempo_int_cir_dias":
        return


    # -----------------------------------------------------
    # NUMÉRICO
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
    # CATEGÓRICO
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

        desfecho_real = (
            "Prolongada"
        )

    else:

        desfecho_real = (
            "Não prolongada"
        )


    if (
        classificacao_prevista
        == "Alto risco"
        and
        desfecho_real
        == "Prolongada"
    ):

        return (
            desfecho_real,
            "VP",
            "Modelo acertou",
        )


    elif (
        classificacao_prevista
        == "Baixo risco"
        and
        desfecho_real
        == "Não prolongada"
    ):

        return (
            desfecho_real,
            "VN",
            "Modelo acertou",
        )


    elif (
        classificacao_prevista
        == "Alto risco"
        and
        desfecho_real
        == "Não prolongada"
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
# MÉTRICAS
# =========================================================

def dividir_seguro(
    numerador,
    denominador,
):

    if denominador == 0:
        return None

    return (
        numerador
        / denominador
    )


def mostrar_percentual(
    valor
):

    if valor is None:
        return "—"

    return f"{valor:.1%}"


# =========================================================
# IDENTIFICAR VARIÁVEL ORIGINAL DO ONE-HOT
# =========================================================

def identificar_variavel_original(
    feature_transformada
):

    nome = str(
        feature_transformada
    )

    nome = (
        nome
        .replace("num__", "")
        .replace("cat__", "")
        .replace("nom__", "")
    )


    # Ordenação por tamanho evita confusão
    # entre nomes semelhantes
    for original in sorted(
        predictors,
        key=len,
        reverse=True,
    ):

        if nome == original:

            return original

        if nome.startswith(
            original + "_"
        ):

            return original


    return nome


# =========================================================
# SHAP AGREGADO POR VARIÁVEL CLÍNICA
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
            "O modelo salvo não possui pipeline compatível "
            "com a explicação SHAP."
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
    # MESMO PRÉ-PROCESSAMENTO DO MODELO
    # -----------------------------------------------------

    X_novo = (
        novo_paciente[
            predictors
        ]
        .copy()
    )


    X_proc = (
        preprocessador
        .transform(
            X_novo
        )
    )


    if hasattr(
        X_proc,
        "toarray",
    ):

        X_proc = (
            X_proc
            .toarray()
        )


    nomes_features = (
        preprocessador
        .get_feature_names_out()
    )


    nomes_features_limpos = [

        str(nome)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("nom__", "")

        for nome
        in nomes_features
    ]


    X_proc_df = pd.DataFrame(
        X_proc,
        columns=nomes_features_limpos,
    )


    # -----------------------------------------------------
    # COMPATIBILIDADE XGBOOST
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
    # SHAP
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
    # TABELA DAS FEATURES TRANSFORMADAS
    # -----------------------------------------------------

    tabela_transformada = (
        pd.DataFrame(
            {
                "feature_transformada":
                    nomes_features_limpos,

                "valor_shap":
                    valores_shap,
            }
        )
    )


    # -----------------------------------------------------
    # IDENTIFICA VARIÁVEL CLÍNICA ORIGINAL
    # -----------------------------------------------------

    tabela_transformada[
        "feature_original"
    ] = (
        tabela_transformada[
            "feature_transformada"
        ]
        .apply(
            identificar_variavel_original
        )
    )


    # -----------------------------------------------------
    # AGREGA TODOS OS DUMMIES DA MESMA VARIÁVEL
    # -----------------------------------------------------

    tabela_agregada = (
        tabela_transformada
        .groupby(
            "feature_original",
            as_index=False,
        )[
            "valor_shap"
        ]
        .sum()
    )


    # -----------------------------------------------------
    # NOME CLÍNICO
    # -----------------------------------------------------

    tabela_agregada[
        "variavel"
    ] = (
        tabela_agregada[
            "feature_original"
        ]
        .apply(
            lambda x:
                nomes_clinicos.get(
                    x,
                    str(x)
                    .replace("_", " ")
                    .capitalize(),
                )
        )
    )


    # -----------------------------------------------------
    # VALOR REAL INFORMADO PELO USUÁRIO
    # -----------------------------------------------------

    valores_originais = (
        novo_paciente
        .iloc[0]
        .to_dict()
    )


    tabela_agregada[
        "valor_informado"
    ] = (
        tabela_agregada[
            "feature_original"
        ]
        .apply(
            lambda x:
                formatar_valor_paciente(
                    x,
                    valores_originais.get(
                        x
                    ),
                )
        )
    )


    tabela_agregada[
        "impacto_absoluto"
    ] = (
        tabela_agregada[
            "valor_shap"
        ]
        .abs()
    )


    tabela_agregada = (
        tabela_agregada
        .sort_values(
            "impacto_absoluto",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    return tabela_agregada


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
    aba_desempenho,
) = st.tabs(
    [
        "🔎 Nova predição",
        "📋 Auditoria",
        "🧠 Entenda a decisão",
        "📊 Desempenho do modelo",
    ]
)


# =========================================================
# ABA 1 — NOVA PREDIÇÃO
# =========================================================

with aba_predicao:


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


    # -----------------------------------------------------
    # IDENTIFICAÇÃO
    # -----------------------------------------------------

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

            data_internacao = st.date_input(
                "Data da internação",
                value=None,
                format="DD/MM/YYYY",
                key="data_internacao",
            )


        with col3:

            data_cirurgia = st.date_input(
                "Data da cirurgia",
                value=None,
                format="DD/MM/YYYY",
                key="data_cirurgia",
            )


        tempo_int_cir_dias = None


        if (
            data_internacao is not None
            and
            data_cirurgia is not None
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


    # -----------------------------------------------------
    # DADOS PARA PREDIÇÃO
    # -----------------------------------------------------

    st.markdown(
        "## Dados para predição"
    )


    linha1_col1, linha1_col2 = (
        st.columns(2)
    )


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


    # -----------------------------------------------------
    # OUTRAS VARIÁVEIS
    # -----------------------------------------------------

    faltantes = [
        feature
        for feature in predictors
        if (
            feature not in valores
            and
            feature
            != "tempo_int_cir_dias"
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


    # -----------------------------------------------------
    # PREDIÇÃO
    # -----------------------------------------------------

    st.write("")


    calcular = st.button(
        "🧠 Calcular risco de internação prolongada",
        type="primary",
        use_container_width=True,
    )


    if calcular:


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


        valores[
            "tempo_int_cir_dias"
        ] = int(
            tempo_int_cir_dias
        )


        # -------------------------------------------------
        # CAMPOS PENDENTES
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

                campos_faltantes.append(
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


        if campos_faltantes:

            st.error(
                "Preencha todos os campos "
                "antes de calcular a predição."
            )


            for campo in campos_faltantes:

                st.write(
                    f"- {campo}"
                )


            st.stop()


        # -------------------------------------------------
        # DATAFRAME
        # -------------------------------------------------

        novo_paciente = pd.DataFrame(
            [valores],
            columns=predictors,
        )


        # -------------------------------------------------
        # PROBABILIDADE
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

            st.exception(
                exc
            )

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
        # REGISTRO SUPABASE
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

            st.exception(
                exc
            )

            st.stop()


        # -------------------------------------------------
        # GUARDA PARA EXPLICAÇÃO SHAP
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


        # -------------------------------------------------
        # RESULTADO
        # -------------------------------------------------

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
            "para visualizar a explicação individual."
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

                st.exception(
                    exc
                )


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


        # -------------------------------------------------
        # CASO AUDITADO
        # -------------------------------------------------

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


        # -------------------------------------------------
        # REGISTRAR ALTA
        # -------------------------------------------------

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


                if data_alta is not None:


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
                            dias_reais,
                        )


                        atualizacao = {

                            "data_alta":
                                data_alta.isoformat(),

                            "dias_reais_internacao":
                                dias_reais,

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


                            st.success(
                                "Alta e auditoria "
                                "registradas com sucesso."
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

                            st.exception(
                                exc
                            )


# =========================================================
# ABA 3 — ENTENDA A DECISÃO
# =========================================================

with aba_explicacao:


    st.markdown(
        "## 🧠 Entenda a decisão"
    )


    ultima_predicao = (
        st.session_state.get(
            "ultima_predicao"
        )
    )


    if not ultima_predicao:

        st.info(
            "Realize uma nova predição para visualizar "
            "a explicação individual do modelo."
        )


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
        # RESULTADO
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


        st.divider()


        # =================================================
        # SHAP
        # =================================================

        st.markdown(
            "### 🔍 Como o modelo chegou a esta previsão?"
        )


        st.write(
            "O SHAP estima a contribuição das variáveis "
            "para esta predição individual. Nesta tela, "
            "as contribuições das categorias criadas pelo "
            "pré-processamento foram reunidas novamente "
            "por variável clínica original."
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


            # =================================================
            # MESMAS VARIÁVEIS PARA GRÁFICO E QUADROS
            # =================================================

            top_shap = (
                tabela_shap
                .head(10)
                .copy()
            )


            aumentam = (
                top_shap[
                    top_shap[
                        "valor_shap"
                    ] > 0
                ]
                .sort_values(
                    "valor_shap",
                    ascending=False,
                )
            )


            reduzem = (
                top_shap[
                    top_shap[
                        "valor_shap"
                    ] < 0
                ]
                .sort_values(
                    "valor_shap",
                    ascending=True,
                )
            )


            # -------------------------------------------------
            # QUADROS
            # -------------------------------------------------

            col_a, col_r = (
                st.columns(2)
            )


            with col_a:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### ⬆️ Contribuíram para maior risco estimado"
                    )


                    if aumentam.empty:

                        st.caption(
                            "Nenhuma das principais variáveis "
                            "apresentou contribuição positiva."
                        )


                    else:

                        for _, linha in aumentam.iterrows():

                            st.write(
                                f"• **{linha['variavel']}**: "
                                f"{linha['valor_informado']}"
                            )


            with col_r:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "#### ⬇️ Contribuíram para menor risco estimado"
                    )


                    if reduzem.empty:

                        st.caption(
                            "Nenhuma das principais variáveis "
                            "apresentou contribuição negativa."
                        )


                    else:

                        for _, linha in reduzem.iterrows():

                            st.write(
                                f"• **{linha['variavel']}**: "
                                f"{linha['valor_informado']}"
                            )


            # =================================================
            # GRÁFICO — EXATAMENTE O MESMO TOP 10
            # =================================================

            st.markdown(
                "### Principais contribuições para esta predição"
            )


            top_plot = (
                top_shap
                .sort_values(
                    "valor_shap",
                    ascending=True,
                )
                .copy()
            )


            labels_plot = [

                (
                    f"{linha['variavel']} "
                    f"({linha['valor_informado']})"
                )

                for _, linha
                in top_plot.iterrows()
            ]


            fig, ax = plt.subplots(
                figsize=(9, 6)
            )


            ax.barh(
                labels_plot,
                top_plot[
                    "valor_shap"
                ],
            )


            ax.axvline(
                0,
                linewidth=1,
            )


            ax.set_xlabel(
                "Valor SHAP agregado"
            )


            ax.set_ylabel(
                ""
            )


            ax.set_title(
                "Contribuição das variáveis clínicas para a predição"
            )


            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True,
            )


            plt.close(
                fig
            )


            st.caption(
                "Valores à direita de zero contribuíram "
                "para maior risco estimado. Valores à esquerda "
                "contribuíram para menor risco estimado."
            )


            st.warning(
                "A explicação SHAP descreve como o modelo "
                "construiu esta predição específica. "
                "Ela não demonstra que uma variável cause "
                "internação prolongada."
            )


            # -------------------------------------------------
            # TABELA COMPLETA
            # -------------------------------------------------

            with st.expander(
                "Ver contribuições SHAP por variável"
            ):


                tabela_exibicao = (
                    tabela_shap[
                        [
                            "variavel",
                            "valor_informado",
                            "valor_shap",
                        ]
                    ]
                    .copy()
                )


                tabela_exibicao.columns = [
                    "Variável",
                    "Valor informado",
                    "Valor SHAP agregado",
                ]


                st.dataframe(
                    tabela_exibicao,
                    use_container_width=True,
                    hide_index=True,
                )


        except Exception as exc:

            st.warning(
                "A predição foi realizada normalmente, "
                "mas não foi possível gerar a explicação SHAP."
            )


            with st.expander(
                "Detalhes técnicos"
            ):

                st.exception(
                    exc
                )


        # =================================================
        # DCA
        # =================================================

        st.divider()


        st.markdown(
            "## 📈 Utilidade clínica da decisão"
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
                    if sensibilidade_meta is not None
                    else "80%"
                ),
            )


        st.info(
            "Na amostra de desenvolvimento, o modelo "
            "apresentou benefício líquido superior às "
            "estratégias de tratar todos e tratar ninguém "
            "em uma faixa contínua de thresholds entre "
            "**21% e 80%**."
        )


        st.write(
            "Essa faixa **não significa que probabilidades "
            "entre 21% e 80% sejam mais precisas**. "
            "Ela representa uma faixa de limiares de decisão "
            "na qual o uso do modelo apresentou benefício "
            "líquido potencial."
        )


        st.write(
            f"O threshold operacional de **{threshold:.0%}** "
            "encontra-se dentro dessa faixa."
        )


# =========================================================
# ABA 4 — DESEMPENHO
# =========================================================

with aba_desempenho:


    st.markdown(
        "## 📊 Desempenho do modelo"
    )


    st.write(
        "As métricas abaixo são calculadas apenas "
        "com pacientes cujo desfecho já foi registrado."
    )


    if st.button(
        "🔄 Atualizar indicadores"
    ):

        st.rerun()


    try:

        registros_completos = (
            carregar_todos_registros()
        )


        df_completo = (
            pd.DataFrame(
                registros_completos
            )
        )


    except Exception as exc:

        st.error(
            "Não foi possível consultar "
            "a base de auditoria."
        )

        st.exception(
            exc
        )

        df_completo = (
            pd.DataFrame()
        )


    if df_completo.empty:

        st.info(
            "Ainda não existem predições registradas."
        )


    else:


        # -------------------------------------------------
        # AUDITADOS
        # -------------------------------------------------

        if (
            "desfecho_real"
            in df_completo.columns
        ):

            mascara_auditados = (
                df_completo[
                    "desfecho_real"
                ]
                .notna()
                &
                (
                    df_completo[
                        "desfecho_real"
                    ]
                    .astype(str)
                    .str.strip()
                    != ""
                )
            )


        else:

            mascara_auditados = pd.Series(
                False,
                index=df_completo.index,
            )


        df_auditados = (
            df_completo[
                mascara_auditados
            ]
            .copy()
        )


        total_predicoes = len(
            df_completo
        )


        total_auditados = len(
            df_auditados
        )


        total_pendentes = (
            total_predicoes
            - total_auditados
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "Total de predições",
                total_predicoes,
            )


        with col2:

            st.metric(
                "Casos auditados",
                total_auditados,
            )


        with col3:

            st.metric(
                "Pendentes de desfecho",
                total_pendentes,
            )


        if not df_auditados.empty:


            tipos = (
                df_auditados[
                    "tipo_resultado"
                ]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
            )


            vp = int(
                (tipos == "VP").sum()
            )

            vn = int(
                (tipos == "VN").sum()
            )

            fp = int(
                (tipos == "FP").sum()
            )

            fn = int(
                (tipos == "FN").sum()
            )


            st.markdown(
                "### Matriz de classificação"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "VP",
                    vp,
                )


            with col2:

                st.metric(
                    "VN",
                    vn,
                )


            with col3:

                st.metric(
                    "FP",
                    fp,
                )


            with col4:

                st.metric(
                    "FN",
                    fn,
                )


            # -------------------------------------------------
            # MÉTRICAS
            # -------------------------------------------------

            sensibilidade = dividir_seguro(
                vp,
                vp + fn,
            )


            especificidade = dividir_seguro(
                vn,
                vn + fp,
            )


            vpp = dividir_seguro(
                vp,
                vp + fp,
            )


            vpn = dividir_seguro(
                vn,
                vn + fn,
            )


            acuracia = dividir_seguro(
                vp + vn,
                vp + vn + fp + fn,
            )


            st.markdown(
                "### Indicadores de desempenho"
            )


            col1, col2, col3 = (
                st.columns(3)
            )


            with col1:

                st.metric(
                    "Sensibilidade",
                    mostrar_percentual(
                        sensibilidade
                    ),
                )


            with col2:

                st.metric(
                    "Especificidade",
                    mostrar_percentual(
                        especificidade
                    ),
                )


            with col3:

                st.metric(
                    "Acurácia",
                    mostrar_percentual(
                        acuracia
                    ),
                )


            col4, col5 = (
                st.columns(2)
            )


            with col4:

                st.metric(
                    "VPP",
                    mostrar_percentual(
                        vpp
                    ),
                )


            with col5:

                st.metric(
                    "VPN",
                    mostrar_percentual(
                        vpn
                    ),
                )


            # -------------------------------------------------
            # MATRIZ DE CONFUSÃO
            # -------------------------------------------------

            matriz = pd.DataFrame(
                [
                    [
                        vn,
                        fp,
                    ],
                    [
                        fn,
                        vp,
                    ],
                ],
                index=[
                    "Real: Não prolongada",
                    "Real: Prolongada",
                ],
                columns=[
                    "Previsto: Baixo risco",
                    "Previsto: Alto risco",
                ],
            )


            st.markdown(
                "### Matriz de confusão"
            )


            st.dataframe(
                matriz,
                use_container_width=True,
            )


        else:

            st.warning(
                "Ainda não há pacientes com "
                "desfecho registrado."
            )


        # =================================================
        # CSV COMPLETO
        # =================================================

        st.divider()


        st.markdown(
            "## ⬇️ Exportar base completa"
        )


        st.warning(
            "O arquivo contém prontuário e demais dados "
            "registrados. O acesso deverá ser restrito "
            "aos usuários autorizados."
        )


        colunas_prioritarias = [
            "id",
            "id_predicao",
            "prontuario",
            "data_predicao",
            "data_internacao",
            "data_cirurgia",
        ]


        colunas_prioritarias += [
            feature
            for feature in predictors
            if feature
            not in colunas_prioritarias
        ]


        colunas_prioritarias += [
            "probabilidade",
            "threshold",
            "classificacao_prevista",
            "modelo",
            "data_alta",
            "dias_reais_internacao",
            "desfecho_real",
            "tipo_resultado",
            "predicao",
            "data_registro_desfecho",
        ]


        colunas_existentes = [
            coluna
            for coluna in colunas_prioritarias
            if coluna in df_completo.columns
        ]


        outras_colunas = [
            coluna
            for coluna in df_completo.columns
            if coluna not in colunas_existentes
        ]


        df_exportacao = (
            df_completo[
                colunas_existentes
                + outras_colunas
            ]
            .copy()
        )


        csv_completo = (
            df_exportacao
            .to_csv(
                index=False,
                sep=";",
            )
            .encode(
                "utf-8-sig"
            )
        )


        data_arquivo = (
            datetime.now()
            .strftime(
                "%Y%m%d"
            )
        )


        st.download_button(
            label="⬇️ Exportar base completa em CSV",
            data=csv_completo,
            file_name=(
                f"auditoria_modelo_ccr_"
                f"{data_arquivo}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
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
        "é calculado automaticamente."
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

Os valores SHAP das categorias produzidas pelo pré-processamento
são agregados novamente por variável clínica original.

Valores positivos contribuem para maior risco estimado.

Valores negativos contribuem para menor risco estimado.

A explicação representa o comportamento do modelo e não causalidade.
        """
    )
