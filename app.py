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
    page_title="CCR | Risco de internação prolongada",
    page_icon="🏥",
    layout="wide",
)

BUNDLE_PATH = Path(__file__).with_name(
    "27_deployment_bundle.joblib"
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
# VERIFICAÇÃO DO THRESHOLD
# =========================================================

if threshold is None:

    st.error(
        "O bundle não possui threshold operacional definido."
    )

    st.stop()

# =========================================================
# CATEGORIAS AMIGÁVEIS
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

    texto = str(valor)

    if texto in mapa_exibicao:
        return mapa_exibicao[texto]

    return (
        texto
        .replace("_", " ")
        .capitalize()
    )

# =========================================================
# FUNÇÃO DOS CAMPOS
# =========================================================

valores = {}


def criar_campo(feature):

    meta = schema.get(feature, {})
    label = meta.get(
        "label",
        feature,
    )

    # -----------------------------------------------------
    # NUMÉRICAS
    # -----------------------------------------------------

    if meta.get("type") == "numeric":

        min_v = meta.get("min")
        max_v = meta.get("max")
        median_v = meta.get("median")

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
            key=f"pred_{feature}",
        )

    # -----------------------------------------------------
    # CATEGÓRICAS
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
                format_func=formatar_categoria,
                key=f"pred_{feature}",
            )

        else:

            valores[feature] = st.text_input(
                label,
                key=f"pred_{feature}",
            )

    else:

        valores[feature] = st.text_input(
            label,
            key=f"pred_{feature}",
        )


# =========================================================
# FUNÇÃO DE AUDITORIA
# =========================================================

def calcular_auditoria(
    classificacao_prevista,
    dias_reais,
):

    # Desfecho real derivado automaticamente
    if dias_reais > 7:
        desfecho_real = "Prolongada"
    else:
        desfecho_real = "Não prolongada"

    # Verdadeiro positivo
    if (
        classificacao_prevista == "Alto risco"
        and desfecho_real == "Prolongada"
    ):

        tipo_resultado = "VP"
        predicao = "Modelo acertou"

    # Verdadeiro negativo
    elif (
        classificacao_prevista == "Baixo risco"
        and desfecho_real == "Não prolongada"
    ):

        tipo_resultado = "VN"
        predicao = "Modelo acertou"

    # Falso positivo
    elif (
        classificacao_prevista == "Alto risco"
        and desfecho_real == "Não prolongada"
    ):

        tipo_resultado = "FP"
        predicao = "Modelo errou"

    # Falso negativo
    else:

        tipo_resultado = "FN"
        predicao = "Modelo errou"

    return (
        desfecho_real,
        tipo_resultado,
        predicao,
    )


# =========================================================
# CABEÇALHO
# =========================================================

st.title(
    "Predição de internação prolongada"
)

st.caption(
    "Pacientes submetidos à cirurgia "
    "por câncer colorretal"
)

st.info(
    "Protótipo para estimar risco de internação > 7 dias. "
    "As predições são registradas para acompanhamento "
    "e auditoria posterior."
)

# =========================================================
# BARRA LATERAL
# =========================================================

st.sidebar.header("Modelo")

st.sidebar.write(
    f"**Algoritmo:** {family}"
)

st.sidebar.write(
    f"**Threshold operacional:** "
    f"{threshold:.0%}"
)

if sensibilidade_meta is not None:

    st.sidebar.write(
        f"**Meta de sensibilidade:** "
        f"{sensibilidade_meta:.0%}"
    )

st.sidebar.caption(
    bundle.get(
        "threshold_rule",
        "",
    )
)

# =========================================================
# ABAS
# =========================================================

aba_predicao, aba_auditoria = st.tabs(
    [
        "🔎 Predição",
        "📋 Auditoria do modelo",
    ]
)

# =========================================================
# ABA 1 - PREDIÇÃO
# =========================================================

with aba_predicao:

    st.subheader(
        "Identificação do caso"
    )

    id_caso = st.text_input(
        "ID pseudonimizado do caso",
        placeholder="Ex.: CCR-0001",
        help=(
            "Utilize um código do estudo. "
            "Não informe nome, CPF ou prontuário."
        ),
    )

    st.subheader(
        "Dados do paciente"
    )

    # -----------------------------------------------------
    # DADOS DEMOGRÁFICOS
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 👤 Dados demográficos"
        )

        col1, col2 = st.columns(2)

        with col1:

            if "sexo_int" in predictors:
                criar_campo("sexo_int")

            if "idade_anos_diag" in predictors:
                criar_campo(
                    "idade_anos_diag"
                )

        with col2:

            if (
                "f_idade_anos_int"
                in predictors
            ):

                criar_campo(
                    "f_idade_anos_int"
                )

    # -----------------------------------------------------
    # CLÍNICA / ONCOLÓGICA
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🩺 Condição clínica e oncológica"
        )

        col1, col2 = st.columns(2)

        with col1:

            if "f_asa" in predictors:
                criar_campo("f_asa")

            if "f_estagio" in predictors:
                criar_campo(
                    "f_estagio"
                )

        with col2:

            if "f_localizacao" in predictors:
                criar_campo(
                    "f_localizacao"
                )

            if (
                "f_neoadjuvancia"
                in predictors
            ):

                criar_campo(
                    "f_neoadjuvancia"
                )

    # -----------------------------------------------------
    # CIRÚRGICO
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🏥 Procedimento cirúrgico"
        )

        col1, col2 = st.columns(2)

        with col1:

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

        with col2:

            if (
                "num_orgaos_envolvidos"
                in predictors
            ):

                criar_campo(
                    "num_orgaos_envolvidos"
                )

            if "urgencia" in predictors:
                criar_campo(
                    "urgencia"
                )

    # -----------------------------------------------------
    # INTERNAÇÃO / UTI
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🛏️ Internação e cuidados intensivos"
        )

        col1, col2 = st.columns(2)

        with col1:

            if (
                "tempo_int_cir_dias"
                in predictors
            ):

                criar_campo(
                    "tempo_int_cir_dias"
                )

        with col2:

            if "uti" in predictors:
                criar_campo(
                    "uti"
                )

    # -----------------------------------------------------
    # GARANTIA DE TODOS OS PREDITORES
    # -----------------------------------------------------

    faltantes = [
        feature
        for feature in predictors
        if feature not in valores
    ]

    if faltantes:

        with st.expander(
            "Outras variáveis"
        ):

            for feature in faltantes:
                criar_campo(feature)

    # -----------------------------------------------------
    # CALCULAR
    # -----------------------------------------------------

    st.divider()

    calcular = st.button(
        "Calcular risco de internação prolongada",
        type="primary",
        use_container_width=True,
    )

    if calcular:

        if not id_caso.strip():

            st.warning(
                "Informe o ID pseudonimizado "
                "do caso antes de calcular."
            )

            st.stop()

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
                "Não foi possível calcular "
                "a predição."
            )

            st.exception(exc)
            st.stop()

        if prob >= threshold:

            classificacao_prevista = (
                "Alto risco"
            )

        else:

            classificacao_prevista = (
                "Baixo risco"
            )

        # -------------------------------------------------
        # ID ÚNICO DA PREDIÇÃO
        # -------------------------------------------------

        id_predicao = str(uuid4())

        # -------------------------------------------------
        # REGISTRO NO SUPABASE
        # -------------------------------------------------

        registro = {

            "id_predicao":
                id_predicao,

            "id_caso":
                id_caso.strip(),

            "probabilidade":
                prob,

            "threshold":
                float(threshold),

            "classificacao_prevista":
                classificacao_prevista,

            "modelo":
                str(family),
        }

        # Adiciona os 13 preditores
        for feature in predictors:

            valor = valores.get(feature)

            meta = schema.get(
                feature,
                {},
            )

            if (
                meta.get("type")
                == "numeric"
            ):

                registro[feature] = (
                    int(valor)
                    if valor is not None
                    else None
                )

            else:

                registro[feature] = (
                    str(valor)
                    if valor is not None
                    else None
                )

        try:

            supabase.table(
                "auditoria_predicoes"
            ).insert(
                registro
            ).execute()

        except Exception as exc:

            st.error(
                "A predição foi calculada, "
                "mas não foi possível registrá-la "
                "no banco de auditoria."
            )

            st.exception(exc)
            st.stop()

        # -------------------------------------------------
        # RESULTADO
        # -------------------------------------------------

        st.markdown(
            "## Resultado"
        )

        with st.container(
            border=True
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Probabilidade de internação > 7 dias",
                    f"{prob:.1%}",
                )

            with col2:

                st.metric(
                    "Classificação",
                    classificacao_prevista.upper(),
                )

            st.progress(
                min(
                    max(prob, 0.0),
                    1.0,
                )
            )

            if (
                classificacao_prevista
                == "Alto risco"
            ):

                st.error(
                    f"⚠️ **Alto risco.** "
                    f"Probabilidade {prob:.1%} "
                    f"≥ threshold "
                    f"{threshold:.0%}."
                )

            else:

                st.success(
                    f"✅ **Baixo risco.** "
                    f"Probabilidade {prob:.1%} "
                    f"< threshold "
                    f"{threshold:.0%}."
                )

        st.success(
            "Predição registrada com sucesso "
            "no banco de auditoria."
        )

        st.write(
            "**ID da predição:**"
        )

        st.code(
            id_predicao
        )

        st.caption(
            "Esse identificador permitirá "
            "vincular posteriormente o desfecho "
            "real à predição realizada."
        )

# =========================================================
# ABA 2 - AUDITORIA
# =========================================================

with aba_auditoria:

    st.subheader(
        "Registrar desfecho real"
    )

    st.write(
        "Após a alta hospitalar, informe o "
        "ID do caso para localizar a predição "
        "e registrar os dias reais de internação."
    )

    id_busca = st.text_input(
        "ID pseudonimizado do caso",
        key="auditoria_id_caso",
        placeholder="Ex.: CCR-0001",
    )

    buscar = st.button(
        "Buscar predição",
        key="buscar_predicao",
    )

    if buscar:

        if not id_busca.strip():

            st.warning(
                "Informe o ID do caso."
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
                        "id_caso",
                        id_busca.strip(),
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
                        "para este ID."
                    )

                else:

                    # Guarda na sessão
                    st.session_state[
                        "registro_auditoria"
                    ] = registros[0]

            except Exception as exc:

                st.error(
                    "Erro ao consultar "
                    "o banco de auditoria."
                )

                st.exception(exc)

    # -----------------------------------------------------
    # REGISTRO ENCONTRADO
    # -----------------------------------------------------

    registro_auditoria = (
        st.session_state.get(
            "registro_auditoria"
        )
    )

    if registro_auditoria:

        st.divider()

        st.markdown(
            "### Predição localizada"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Probabilidade",
                f"{float(registro_auditoria['probabilidade']):.1%}",
            )

        with col2:

            st.metric(
                "Classificação",
                registro_auditoria[
                    "classificacao_prevista"
                ],
            )

        with col3:

            st.metric(
                "Threshold",
                f"{float(registro_auditoria['threshold']):.0%}",
            )

        st.write(
            f"**ID da predição:** "
            f"{registro_auditoria['id_predicao']}"
        )

        # -------------------------------------------------
        # SE JÁ FOI AUDITADO
        # -------------------------------------------------

        if (
            registro_auditoria.get(
                "desfecho_real"
            )
        ):

            st.info(
                "Este caso já possui "
                "desfecho registrado."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Desfecho real",
                    registro_auditoria[
                        "desfecho_real"
                    ],
                )

            with col2:

                st.metric(
                    "Resultado",
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
        # REGISTRAR DESFECHO
        # -------------------------------------------------

        else:

            dias_reais = st.number_input(
                "Dias reais de internação após a cirurgia",
                min_value=0,
                max_value=365,
                value=7,
                step=1,
                format="%d",
            )

            salvar_desfecho = st.button(
                "Registrar desfecho e auditar modelo",
                type="primary",
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

                    st.success(
                        "Desfecho registrado "
                        "com sucesso."
                    )

                    st.markdown(
                        "### Resultado da auditoria"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:

                        st.metric(
                            "Desfecho real",
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

                    # Limpa cache de busca
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

# =========================================================
# SOBRE
# =========================================================

st.divider()

with st.expander(
    "Sobre o sistema de auditoria"
):

    st.write(
        "Cada predição é registrada antes que "
        "o desfecho real seja conhecido."
    )

    st.write(
        "Após a alta, os dias reais de internação "
        "são informados. A aplicação deriva "
        "automaticamente o desfecho."
    )

    st.markdown(
        """
**Regras de auditoria**

- **VP:** previu alto risco e houve internação prolongada.
- **VN:** previu baixo risco e não houve internação prolongada.
- **FP:** previu alto risco e não houve internação prolongada.
- **FN:** previu baixo risco e houve internação prolongada.
        """
    )

    st.write(
        "VP e VN → **Modelo acertou**"
    )

    st.write(
        "FP e FN → **Modelo errou**"
    )
