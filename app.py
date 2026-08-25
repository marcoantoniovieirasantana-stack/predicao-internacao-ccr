import streamlit as st
import joblib
import pandas as pd
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
# LIMPEZA AO INICIAR UMA NOVA SESSÃO
# =========================================================

if "app_inicializado" not in st.session_state:

    chaves_para_limpar = [
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
            ]
        )
    ]

    for chave in chaves_para_limpar:
        st.session_state.pop(
            chave,
            None,
        )

    st.session_state[
        "app_inicializado"
    ] = True


# =========================================================
# FUNÇÃO NOVO CADASTRO
# =========================================================

def limpar_novo_cadastro():

    chaves_para_limpar = [
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
            ]
        )
    ]

    for chave in chaves_para_limpar:

        st.session_state.pop(
            chave,
            None,
        )


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

    h1 {
        font-size: 2.1rem !important;
        margin-bottom: 0.15rem !important;
    }

    h2 {
        margin-top: 1.4rem !important;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    .app-header {
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.04);
        margin-bottom: 1rem;
    }

    .app-header-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .app-header-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-top: 0.25rem;
    }

    .small-label {
        font-size: 0.85rem;
        opacity: 0.65;
        margin-bottom: 0.15rem;
    }

    .derived-value {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        background: rgba(128, 128, 128, 0.04);
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .result-low {
        padding: 1.3rem;
        border-radius: 14px;
        border: 1px solid rgba(30, 150, 80, 0.35);
        background: rgba(30, 150, 80, 0.08);
        text-align: center;
        margin-top: 0.7rem;
    }

    .result-high {
        padding: 1.3rem;
        border-radius: 14px;
        border: 1px solid rgba(190, 40, 40, 0.35);
        background: rgba(190, 40, 40, 0.08);
        text-align: center;
        margin-top: 0.7rem;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .result-prob {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .audit-step {
        font-size: 0.9rem;
        font-weight: 700;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CARREGAMENTO DO MODELO
# =========================================================

@st.cache_resource
def carregar_bundle():

    return joblib.load(
        BUNDLE_PATH
    )


bundle = carregar_bundle()

model = bundle["model"]

predictors = bundle[
    "predictors"
]

schema = bundle.get(
    "ui_schema",
    {},
)

threshold = bundle.get(
    "clinical_threshold"
)

family = bundle.get(
    "family",
    "Modelo",
)

sensibilidade_meta = bundle.get(
    "min_sensitivity_target"
)


# =========================================================
# CONEXÃO COM SUPABASE
# =========================================================

@st.cache_resource
def conectar_supabase():

    url = st.secrets[
        "supabase"
    ]["url"]

    key = st.secrets[
        "supabase"
    ]["key"]

    return create_client(
        url,
        key,
    )


try:

    supabase = (
        conectar_supabase()
    )

except Exception as exc:

    st.error(
        "Não foi possível conectar "
        "ao banco de auditoria."
    )

    st.exception(
        exc
    )

    st.stop()


# =========================================================
# VALIDAÇÃO DO THRESHOLD
# =========================================================

if threshold is None:

    st.error(
        "O bundle não possui threshold "
        "operacional definido."
    )

    st.stop()


# =========================================================
# CATEGORIAS AMIGÁVEIS
# =========================================================

mapa_exibicao = {

    "s": "Sim",
    "n": "Não",

    "colon_direito":
        "Cólon direito",

    "colon_esquerdo":
        "Cólon esquerdo",

    "reto_inferior":
        "Reto inferior",

    "reto_medio":
        "Reto médio",

    "retossigmoide":
        "Retossigmoide",

    "sincronico":
        "Sincrônico",

    "convencional":
        "Convencional",

    "laparoscopica":
        "Laparoscópica",
}


def formatar_categoria(
    valor
):

    if valor is None:
        return ""

    texto = str(
        valor
    )

    if texto in mapa_exibicao:

        return mapa_exibicao[
            texto
        ]

    return (
        texto
        .replace(
            "_",
            " ",
        )
        .capitalize()
    )


# =========================================================
# CAMPOS DO MODELO
# =========================================================

valores = {}


def criar_campo(
    feature
):

    meta = schema.get(
        feature,
        {},
    )

    label = meta.get(
        "label",
        feature,
    )


    # -----------------------------------------------------
    # TEMPO ENTRE INTERNAÇÃO E CIRURGIA
    # É DERIVADO DAS DATAS
    # -----------------------------------------------------

    if (
        feature
        == "tempo_int_cir_dias"
    ):

        return


    # -----------------------------------------------------
    # VARIÁVEL NUMÉRICA
    # -----------------------------------------------------

    if (
        meta.get("type")
        == "numeric"
    ):

        min_v = meta.get(
            "min"
        )

        max_v = meta.get(
            "max"
        )

        help_txt = None


        if (
            min_v is not None
            and max_v is not None
        ):

            help_txt = (
                "Faixa observada no banco "
                "de desenvolvimento: "
                f"{int(round(min_v))} a "
                f"{int(round(max_v))}."
            )


        valores[
            feature
        ] = st.number_input(

            label,

            min_value=(
                int(
                    round(min_v)
                )
                if min_v
                is not None
                else None
            ),

            max_value=(
                int(
                    round(max_v)
                )
                if max_v
                is not None
                else None
            ),

            value=None,

            step=1,

            format="%d",

            placeholder=(
                "Informe o valor"
            ),

            help=help_txt,

            key=(
                f"pred_{feature}"
            ),
        )


    # -----------------------------------------------------
    # VARIÁVEL CATEGÓRICA
    # -----------------------------------------------------

    elif (
        meta.get("type")
        == "categorical"
    ):

        options = meta.get(
            "options",
            [],
        )


        if options:

            valores[
                feature
            ] = st.selectbox(

                label,

                options=options,

                index=None,

                placeholder=(
                    "Selecione uma opção"
                ),

                format_func=(
                    formatar_categoria
                ),

                key=(
                    f"pred_{feature}"
                ),
            )

        else:

            valores[
                feature
            ] = st.text_input(

                label,

                value="",

                key=(
                    f"pred_{feature}"
                ),
            )


    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    else:

        valores[
            feature
        ] = st.text_input(

            label,

            value="",

            key=(
                f"pred_{feature}"
            ),
        )


# =========================================================
# FUNÇÃO DE AUDITORIA
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

        tipo_resultado = "VP"

        predicao = (
            "Modelo acertou"
        )


    elif (
        classificacao_prevista
        == "Baixo risco"
        and
        desfecho_real
        == "Não prolongada"
    ):

        tipo_resultado = "VN"

        predicao = (
            "Modelo acertou"
        )


    elif (
        classificacao_prevista
        == "Alto risco"
        and
        desfecho_real
        == "Não prolongada"
    ):

        tipo_resultado = "FP"

        predicao = (
            "Modelo errou"
        )


    else:

        tipo_resultado = "FN"

        predicao = (
            "Modelo errou"
        )


    return (
        desfecho_real,
        tipo_resultado,
        predicao,
    )


# =========================================================
# CABEÇALHO
# =========================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-header-title">
            🏥 Predição de Internação Prolongada — CCR
        </div>

        <div class="app-header-subtitle">
            Apoio à decisão no pós-operatório de pacientes
            submetidos à cirurgia por câncer colorretal
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "Protótipo em desenvolvimento. "
    "A interpretação deve considerar "
    "o desempenho e as limitações do modelo."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## 🧠 Sobre o modelo"
)

st.sidebar.write(
    f"**Algoritmo:** "
    f"{family}"
)

st.sidebar.write(
    "**Desfecho:** "
    "Internação > 7 dias"
)

st.sidebar.write(
    f"**Threshold:** "
    f"{threshold:.0%}"
)


if (
    sensibilidade_meta
    is not None
):

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

aba_predicao, aba_auditoria = st.tabs(
    [
        "🔎 Nova predição",
        "📋 Auditoria",
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

                placeholder=(
                    "Digite o prontuário"
                ),

                key="prontuario",
            )


        with col2:

            data_internacao = (
                st.date_input(

                    "Data da internação",

                    value=None,

                    format=(
                        "DD/MM/YYYY"
                    ),

                    key=(
                        "data_internacao"
                    ),
                )
            )


        with col3:

            data_cirurgia = (
                st.date_input(

                    "Data da cirurgia",

                    value=None,

                    format=(
                        "DD/MM/YYYY"
                    ),

                    key=(
                        "data_cirurgia"
                    ),
                )
            )


        # -------------------------------------------------
        # TEMPO ENTRE INTERNAÇÃO E CIRURGIA
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


                st.markdown(
                    f"""
                    <div class="derived-value">

                        ⏱ Intervalo internação → cirurgia:
                        {tempo_int_cir_dias}
                        {"dia" if tempo_int_cir_dias == 1 else "dias"}

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            else:

                st.error(
                    "A data da cirurgia "
                    "não pode ser anterior "
                    "à data da internação."
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


            if (
                "sexo_int"
                in predictors
            ):

                criar_campo(
                    "sexo_int"
                )


            if (
                "f_idade_anos_int"
                in predictors
            ):

                criar_campo(
                    "f_idade_anos_int"
                )


            if (
                "idade_anos_diag"
                in predictors
            ):

                criar_campo(
                    "idade_anos_diag"
                )


            if (
                "f_asa"
                in predictors
            ):

                criar_campo(
                    "f_asa"
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


            if (
                "f_localizacao"
                in predictors
            ):

                criar_campo(
                    "f_localizacao"
                )


            if (
                "f_estagio"
                in predictors
            ):

                criar_campo(
                    "f_estagio"
                )


            if (
                "f_neoadjuvancia"
                in predictors
            ):

                criar_campo(
                    "f_neoadjuvancia"
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


            if (
                "f_abord_cirurgica"
                in predictors
            ):

                criar_campo(
                    "f_abord_cirurgica"
                )


            if (
                "tempo_cir_min2"
                in predictors
            ):

                criar_campo(
                    "tempo_cir_min2"
                )


            if (
                "num_orgaos_envolvidos"
                in predictors
            ):

                criar_campo(
                    "num_orgaos_envolvidos"
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


            if (
                "uti"
                in predictors
            ):

                criar_campo(
                    "uti"
                )


            if (
                "urgencia"
                in predictors
            ):

                criar_campo(
                    "urgencia"
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
                    "O intervalo internação → cirurgia "
                    "será calculado após o preenchimento "
                    "das duas datas."
                )


    # =====================================================
    # PREDITORES EVENTUALMENTE NÃO MAPEADOS
    # =====================================================

    faltantes = [

        feature

        for feature
        in predictors

        if (
            feature
            not in valores
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


    # =====================================================
    # BOTÃO DE PREDIÇÃO
    # =====================================================

    st.write("")


    calcular = st.button(

        "🧠 Calcular risco de internação prolongada",

        type="primary",

        use_container_width=True,
    )


    # =====================================================
    # EXECUÇÃO DA PREDIÇÃO
    # =====================================================

    if calcular:


        # -------------------------------------------------
        # VALIDAÇÕES DE IDENTIFICAÇÃO
        # -------------------------------------------------

        if not prontuario.strip():

            st.warning(
                "Informe o prontuário."
            )

            st.stop()


        if (
            data_internacao
            is None
        ):

            st.warning(
                "Informe a data da internação."
            )

            st.stop()


        if (
            data_cirurgia
            is None
        ):

            st.warning(
                "Informe a data da cirurgia."
            )

            st.stop()


        if (
            data_cirurgia
            < data_internacao
        ):

            st.error(
                "A data da cirurgia "
                "não pode ser anterior "
                "à data da internação."
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


        # -------------------------------------------------
        # GARANTE VARIÁVEL DERIVADA
        # -------------------------------------------------

        valores[
            "tempo_int_cir_dias"
        ] = int(
            tempo_int_cir_dias
        )


        # -------------------------------------------------
        # VERIFICA TODOS OS PREDITORES
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
                        feature,
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
        # DATAFRAME DO MODELO
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

            st.exception(
                exc
            )

            st.stop()


        # -------------------------------------------------
        # CLASSIFICAÇÃO
        # -------------------------------------------------

        if prob >= threshold:

            classificacao_prevista = (
                "Alto risco"
            )

        else:

            classificacao_prevista = (
                "Baixo risco"
            )


        # -------------------------------------------------
        # ID DA PREDIÇÃO
        # -------------------------------------------------

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
                float(
                    threshold
                ),

            "classificacao_prevista":
                classificacao_prevista,

            "modelo":
                str(
                    family
                ),
        }


        # -------------------------------------------------
        # ADICIONA PREDITORES
        # -------------------------------------------------

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
                ] = int(
                    valor
                )


            else:

                registro[
                    feature
                ] = str(
                    valor
                )


        # -------------------------------------------------
        # SALVA NO SUPABASE
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

            st.exception(
                exc
            )

            st.stop()


        # =================================================
        # RESULTADO
        # =================================================

        st.markdown(
            "## Resultado da predição"
        )


        if (
            classificacao_prevista
            == "Alto risco"
        ):

            st.markdown(
                f"""
                <div class="result-high">

                    <div class="small-label">
                        Probabilidade estimada de internação &gt; 7 dias
                    </div>

                    <div class="result-prob">
                        {prob:.1%}
                    </div>

                    <div class="result-title">
                        🔴 ALTO RISCO
                    </div>

                    <div>
                        Threshold operacional:
                        {threshold:.0%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        else:

            st.markdown(
                f"""
                <div class="result-low">

                    <div class="small-label">
                        Probabilidade estimada de internação &gt; 7 dias
                    </div>

                    <div class="result-prob">
                        {prob:.1%}
                    </div>

                    <div class="result-title">
                        🟢 BAIXO RISCO
                    </div>

                    <div>
                        Threshold operacional:
                        {threshold:.0%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
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


        st.success(
            "Predição registrada "
            "no banco de auditoria."
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
                f"**Intervalo internação → cirurgia:** "
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
        """
        <div class="audit-step">
            Etapa 1 — Localizar paciente
        </div>
        """,
        unsafe_allow_html=True,
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

                    key=(
                        "auditoria_prontuario"
                    ),

                    placeholder=(
                        "Digite o prontuário"
                    ),
                )
            )


        with col_botao:

            st.write("")
            st.write("")

            buscar = st.button(

                "🔍 Buscar",

                key=(
                    "buscar_predicao"
                ),

                use_container_width=True,
            )


    # =====================================================
    # BUSCA
    # =====================================================

    if buscar:


        # Remove resultado antigo antes de nova busca
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
            """
            <div class="audit-step">
                Etapa 2 — Conferir predição
            </div>
            """,
            unsafe_allow_html=True,
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
                    f"**Prontuário:** "
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
                        f"**Internação:** "
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
                        f"**Cirurgia:** "
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
        # CASO JÁ AUDITADO
        # =================================================

        if (
            registro_auditoria.get(
                "desfecho_real"
            )
        ):


            st.markdown(
                """
                <div class="audit-step">
                    Auditoria concluída
                </div>
                """,
                unsafe_allow_html=True,
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
                        f"**Data da alta:** "
                        f"{data_alta_registro.strftime('%d/%m/%Y')}"
                    )


                if (
                    registro_auditoria.get(
                        "dias_reais_internacao"
                    )
                    is not None
                ):

                    st.write(
                        f"**Cirurgia → alta:** "
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
                """
                <div class="audit-step">
                    Etapa 3 — Registrar alta
                </div>
                """,
                unsafe_allow_html=True,
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


                data_alta = (
                    st.date_input(

                        "Data da alta",

                        value=None,

                        min_value=(
                            data_cirurgia_registro
                        ),

                        format=(
                            "DD/MM/YYYY"
                        ),

                        key=(
                            "auditoria_data_alta"
                        ),
                    )
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

                        if dias_reais > 7:

                            st.metric(
                                "Desfecho calculado",
                                "Prolongada",
                            )

                        else:

                            st.metric(
                                "Desfecho calculado",
                                "Não prolongada",
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

                            st.exception(
                                exc
                            )


                else:

                    st.caption(
                        "Informe a data da alta "
                        "para realizar a auditoria."
                    )


# =========================================================
# RODAPÉ
# =========================================================

st.divider()


with st.expander(
    "ℹ️ Informações metodológicas"
):

    st.write(
        "Internação prolongada é definida "
        "como mais de 7 dias entre "
        "a cirurgia e a alta."
    )

    st.write(
        "O intervalo entre internação "
        "e cirurgia é calculado "
        "automaticamente."
    )

    st.write(
        "A data da alta é utilizada "
        "para calcular o tempo real "
        "após a cirurgia."
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
