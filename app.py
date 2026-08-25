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
            ]
        )
    ]

    for chave in chaves:
        st.session_state.pop(chave, None)


# =========================================================
# CSS
# Apenas elementos visuais secundários
# NÃO existe HTML no cabeçalho
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
# CARREGAMENTO DO MODELO
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
# CONEXÃO COM SUPABASE
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
# NOMES AMIGÁVEIS DAS CATEGORIAS
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


# =========================================================
# CRIAÇÃO DOS CAMPOS
# =========================================================

valores = {}


def criar_campo(feature):

    meta = schema.get(feature, {})
    label = meta.get("label", feature)

    # Este campo é calculado automaticamente pelas datas
    if feature == "tempo_int_cir_dias":
        return

    # -----------------------------------------------------
    # VARIÁVEL NUMÉRICA
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
    # VARIÁVEL CATEGÓRICA
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
    # OUTROS TIPOS
    # -----------------------------------------------------

    else:

        valores[feature] = st.text_input(
            label,
            value="",
            key=f"pred_{feature}",
        )


# =========================================================
# FUNÇÃO PARA AUDITORIA
# =========================================================

def calcular_auditoria(
    classificacao_prevista,
    dias_reais,
):

    if dias_reais > 7:
        desfecho_real = "Prolongada"

    else:
        desfecho_real = "Não prolongada"

    # Verdadeiro positivo
    if (
        classificacao_prevista == "Alto risco"
        and desfecho_real == "Prolongada"
    ):

        return (
            desfecho_real,
            "VP",
            "Modelo acertou",
        )

    # Verdadeiro negativo
    elif (
        classificacao_prevista == "Baixo risco"
        and desfecho_real == "Não prolongada"
    ):

        return (
            desfecho_real,
            "VN",
            "Modelo acertou",
        )

    # Falso positivo
    elif (
        classificacao_prevista == "Alto risco"
        and desfecho_real == "Não prolongada"
    ):

        return (
            desfecho_real,
            "FP",
            "Modelo errou",
        )

    # Falso negativo
    else:

        return (
            desfecho_real,
            "FN",
            "Modelo errou",
        )


# =========================================================
# CABEÇALHO
# SOMENTE COMPONENTES NATIVOS DO STREAMLIT
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
# BARRA LATERAL
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

    col_titulo, col_novo = st.columns(
        [4, 1]
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

    with st.container(border=True):

        st.markdown(
            "### 🪪 Identificação do paciente"
        )

        col1, col2, col3 = st.columns(3)

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


        # -------------------------------------------------
        # TEMPO INTERNAÇÃO → CIRURGIA
        # -------------------------------------------------

        tempo_int_cir_dias = None


        if (
            data_internacao is not None
            and data_cirurgia is not None
        ):

            if data_cirurgia >= data_internacao:

                tempo_int_cir_dias = (
                    data_cirurgia
                    - data_internacao
                ).days

                valores[
                    "tempo_int_cir_dias"
                ] = tempo_int_cir_dias

                st.info(
                    f"⏱ Intervalo entre internação e cirurgia: "
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


    # =====================================================
    # PRIMEIRA LINHA
    # =====================================================

    linha1_col1, linha1_col2 = (
        st.columns(2)
    )


    # -----------------------------------------------------
    # DADOS DO PACIENTE
    # -----------------------------------------------------

    with linha1_col1:

        with st.container(border=True):

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
                    criar_campo(feature)


    # -----------------------------------------------------
    # DADOS ONCOLÓGICOS
    # -----------------------------------------------------

    with linha1_col2:

        with st.container(border=True):

            st.markdown(
                "### 🩺 Dados oncológicos"
            )

            for feature in [
                "f_localizacao",
                "f_estagio",
                "f_neoadjuvancia",
            ]:

                if feature in predictors:
                    criar_campo(feature)


    # =====================================================
    # SEGUNDA LINHA
    # =====================================================

    linha2_col1, linha2_col2 = (
        st.columns(2)
    )


    # -----------------------------------------------------
    # PROCEDIMENTO CIRÚRGICO
    # -----------------------------------------------------

    with linha2_col1:

        with st.container(border=True):

            st.markdown(
                "### 🏥 Procedimento cirúrgico"
            )

            for feature in [
                "f_abord_cirurgica",
                "tempo_cir_min2",
                "num_orgaos_envolvidos",
            ]:

                if feature in predictors:
                    criar_campo(feature)


    # -----------------------------------------------------
    # INTERNAÇÃO E CUIDADOS
    # -----------------------------------------------------

    with linha2_col2:

        with st.container(border=True):

            st.markdown(
                "### 🛏️ Internação e cuidados"
            )

            for feature in [
                "uti",
                "urgencia",
            ]:

                if feature in predictors:
                    criar_campo(feature)


            if tempo_int_cir_dias is not None:

                st.metric(
                    "Intervalo internação → cirurgia",
                    f"{tempo_int_cir_dias} dias",
                )

            else:

                st.caption(
                    "O intervalo entre internação e cirurgia "
                    "será calculado automaticamente após "
                    "o preenchimento das duas datas."
                )


    # =====================================================
    # PREDITORES NÃO MAPEADOS
    # =====================================================

    faltantes = [
        feature
        for feature in predictors
        if (
            feature not in valores
            and feature != "tempo_int_cir_dias"
        )
    ]


    if faltantes:

        with st.expander(
            "Outras variáveis"
        ):

            for feature in faltantes:
                criar_campo(feature)


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
    # EXECUÇÃO DA PREDIÇÃO
    # =====================================================

    if calcular:

        # -------------------------------------------------
        # PRONTUÁRIO
        # -------------------------------------------------

        if not prontuario.strip():

            st.warning(
                "Informe o prontuário."
            )

            st.stop()


        # -------------------------------------------------
        # DATA DA INTERNAÇÃO
        # -------------------------------------------------

        if data_internacao is None:

            st.warning(
                "Informe a data da internação."
            )

            st.stop()


        # -------------------------------------------------
        # DATA DA CIRURGIA
        # -------------------------------------------------

        if data_cirurgia is None:

            st.warning(
                "Informe a data da cirurgia."
            )

            st.stop()


        # -------------------------------------------------
        # VALIDAÇÃO DAS DATAS
        # -------------------------------------------------

        if data_cirurgia < data_internacao:

            st.error(
                "A data da cirurgia não pode ser "
                "anterior à data da internação."
            )

            st.stop()


        if tempo_int_cir_dias is None:

            st.error(
                "Não foi possível calcular o intervalo "
                "entre internação e cirurgia."
            )

            st.stop()


        # -------------------------------------------------
        # VARIÁVEL DERIVADA
        # -------------------------------------------------

        valores[
            "tempo_int_cir_dias"
        ] = int(
            tempo_int_cir_dias
        )


        # =================================================
        # VALIDAÇÃO DOS PREDITORES
        # =================================================

        campos_faltantes = []


        for feature in predictors:

            valor = valores.get(
                feature
            )

            if (
                valor is None
                or (
                    isinstance(valor, str)
                    and not valor.strip()
                )
            ):

                label = (
                    schema
                    .get(feature, {})
                    .get("label", feature)
                )

                campos_faltantes.append(
                    label
                )


        if campos_faltantes:

            st.error(
                "Preencha todos os campos antes "
                "de calcular a predição."
            )

            st.write(
                "**Campos pendentes:**"
            )

            for campo in campos_faltantes:

                st.write(
                    f"- {campo}"
                )

            st.stop()


        # =================================================
        # DATAFRAME DO PACIENTE
        # =================================================

        novo_paciente = pd.DataFrame(
            [valores],
            columns=predictors,
        )


        # =================================================
        # PREDIÇÃO
        # =================================================

        try:

            prob = float(
                model.predict_proba(
                    novo_paciente
                )[0, 1]
            )

        except Exception as exc:

            st.error(
                "Não foi possível calcular a predição."
            )

            st.exception(exc)
            st.stop()


        # =================================================
        # CLASSIFICAÇÃO
        # =================================================

        classificacao_prevista = (
            "Alto risco"
            if prob >= threshold
            else "Baixo risco"
        )


        # =================================================
        # ID DA PREDIÇÃO
        # =================================================

        id_predicao = str(
            uuid4()
        )


        # =================================================
        # REGISTRO NO BANCO
        # =================================================

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


        # -------------------------------------------------
        # PREDITORES
        # -------------------------------------------------

        for feature in predictors:

            valor = valores.get(
                feature
            )

            meta = schema.get(
                feature,
                {},
            )

            if meta.get("type") == "numeric":

                registro[
                    feature
                ] = int(valor)

            else:

                registro[
                    feature
                ] = str(valor)


        # =================================================
        # SALVAR NO SUPABASE
        # =================================================

        try:

            supabase.table(
                "auditoria_predicoes"
            ).insert(
                registro
            ).execute()

        except Exception as exc:

            st.error(
                "A predição foi calculada, mas não foi "
                "possível registrá-la no banco de auditoria."
            )

            st.exception(exc)
            st.stop()


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
                max(prob, 0.0),
                1.0,
            )
        )


        if classificacao_prevista == "Alto risco":

            st.error(
                f"🔴 ALTO RISCO — "
                f"probabilidade {prob:.1%} ≥ "
                f"threshold {threshold:.0%}."
            )

        else:

            st.success(
                f"🟢 BAIXO RISCO — "
                f"probabilidade {prob:.1%} < "
                f"threshold {threshold:.0%}."
            )


        st.success(
            "Predição registrada no banco de auditoria."
        )


        with st.expander(
            "Detalhes do registro"
        ):

            st.write(
                f"**Prontuário:** {prontuario}"
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
        "### 1️⃣ Localizar paciente"
    )


    with st.container(border=True):

        col_busca, col_botao = st.columns(
            [3, 1]
        )


        with col_busca:

            prontuario_busca = st.text_input(
                "Prontuário",
                value="",
                key="auditoria_prontuario",
                placeholder="Digite o prontuário",
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
    # BUSCAR PACIENTE
    # =====================================================

    if buscar:

        # Remove resultado anterior
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
                    "Erro ao consultar o banco de auditoria."
                )

                st.exception(exc)


    # =====================================================
    # REGISTRO ENCONTRADO
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


        with st.container(border=True):

            data_cirurgia_registro = None

            col1, col2 = st.columns(2)


            with col1:

                st.write(
                    f"**Prontuário:** "
                    f"{registro_auditoria.get('prontuario', '')}"
                )


                if registro_auditoria.get(
                    "data_internacao"
                ):

                    data_int = pd.to_datetime(
                        registro_auditoria[
                            "data_internacao"
                        ]
                    )

                    st.write(
                        f"**Internação:** "
                        f"{data_int.strftime('%d/%m/%Y')}"
                    )


                if registro_auditoria.get(
                    "data_cirurgia"
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

                met1, met2 = st.columns(2)


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
        # AUDITORIA JÁ REALIZADA
        # =================================================

        if registro_auditoria.get(
            "desfecho_real"
        ):

            st.markdown(
                "### ✅ Auditoria concluída"
            )


            with st.container(border=True):

                if registro_auditoria.get(
                    "data_alta"
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
                        "✅ Modelo acertou a classificação."
                    )

                else:

                    st.error(
                        "❌ Modelo errou a classificação."
                    )


        # =================================================
        # REGISTRAR ALTA
        # =================================================

        else:

            st.markdown(
                "### 3️⃣ Registrar alta"
            )


            with st.container(border=True):

                if data_cirurgia_registro is None:

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


                    col1, col2 = st.columns(2)


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


                    salvar_desfecho = st.button(
                        "✅ Registrar alta e auditar modelo",
                        type="primary",
                        use_container_width=True,
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
                            int(dias_reais),
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
