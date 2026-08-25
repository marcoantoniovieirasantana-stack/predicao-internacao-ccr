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
# FUNÇÃO PARA CRIAR CAMPOS DO MODELO
# =========================================================

valores = {}


def criar_campo(feature):

    meta = schema.get(feature, {})
    label = meta.get(
        "label",
        feature,
    )

    # tempo_int_cir_dias é calculado pelas datas
    if feature == "tempo_int_cir_dias":
        return

    # -----------------------------------------------------
    # VARIÁVEIS QUANTITATIVAS
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
    # VARIÁVEIS CATEGÓRICAS
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

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

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

    if dias_reais > 7:
        desfecho_real = "Prolongada"

    else:
        desfecho_real = "Não prolongada"

    if (
        classificacao_prevista == "Alto risco"
        and desfecho_real == "Prolongada"
    ):

        tipo_resultado = "VP"
        predicao = "Modelo acertou"

    elif (
        classificacao_prevista == "Baixo risco"
        and desfecho_real == "Não prolongada"
    ):

        tipo_resultado = "VN"
        predicao = "Modelo acertou"

    elif (
        classificacao_prevista == "Alto risco"
        and desfecho_real == "Não prolongada"
    ):

        tipo_resultado = "FP"
        predicao = "Modelo errou"

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

    # =====================================================
    # IDENTIFICAÇÃO
    # =====================================================

    st.subheader(
        "Identificação do paciente"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        prontuario = st.text_input(
            "Prontuário",
            placeholder="Digite o prontuário",
        )

    with col2:

        data_internacao = st.date_input(
            "Data da internação",
            format="DD/MM/YYYY",
        )

    with col3:

        data_cirurgia = st.date_input(
            "Data da cirurgia",
            format="DD/MM/YYYY",
        )


    # =====================================================
    # TEMPO ENTRE INTERNAÇÃO E CIRURGIA
    # =====================================================

    if data_cirurgia >= data_internacao:

        tempo_int_cir_dias = (
            data_cirurgia
            - data_internacao
        ).days

        valores[
            "tempo_int_cir_dias"
        ] = tempo_int_cir_dias

        st.info(
            f"**Tempo entre internação e cirurgia:** "
            f"{tempo_int_cir_dias} dia"
            f"{'s' if tempo_int_cir_dias != 1 else ''}."
        )

    else:

        tempo_int_cir_dias = None

        st.error(
            "A data da cirurgia não pode ser "
            "anterior à data da internação."
        )


    # =====================================================
    # DADOS DO PACIENTE
    # =====================================================

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
    # CONDIÇÃO CLÍNICA E ONCOLÓGICA
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
    # PROCEDIMENTO CIRÚRGICO
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
    # INTERNAÇÃO E CUIDADOS INTENSIVOS
    # -----------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🛏️ Internação e cuidados intensivos"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "**Tempo entre internação e cirurgia (dias)**"
            )

            if tempo_int_cir_dias is not None:

                st.metric(
                    "Calculado automaticamente",
                    tempo_int_cir_dias,
                )

            else:

                st.warning(
                    "Revise as datas informadas."
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
        and feature != "tempo_int_cir_dias"
    ]

    if faltantes:

        with st.expander(
            "Outras variáveis"
        ):

            for feature in faltantes:
                criar_campo(feature)


    # =====================================================
    # CALCULAR PREDIÇÃO
    # =====================================================

    st.divider()

    calcular = st.button(
        "Calcular risco de internação prolongada",
        type="primary",
        use_container_width=True,
    )


    if calcular:

        # -------------------------------------------------
        # VALIDAÇÕES
        # -------------------------------------------------

        if not prontuario.strip():

            st.warning(
                "Informe o prontuário antes de calcular."
            )

            st.stop()


        if data_cirurgia < data_internacao:

            st.error(
                "A data da cirurgia não pode ser "
                "anterior à data da internação."
            )

            st.stop()


        if tempo_int_cir_dias is None:

            st.error(
                "Não foi possível calcular o tempo entre "
                "internação e cirurgia."
            )

            st.stop()


        valores[
            "tempo_int_cir_dias"
        ] = int(
            tempo_int_cir_dias
        )


        # -------------------------------------------------
        # DATAFRAME DO MODELO
        # -------------------------------------------------

        novo_paciente = pd.DataFrame(
            [valores],
            columns=predictors,
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
                "Não foi possível calcular a predição."
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

        id_predicao = str(
            uuid4()
        )


        # -------------------------------------------------
        # REGISTRO PARA SUPABASE
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


        # -------------------------------------------------
        # SALVA NO BANCO
        # -------------------------------------------------

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


        # =================================================
        # RESULTADO
        # =================================================

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
            f"**Prontuário:** {prontuario}"
        )

        st.write(
            f"**Data da internação:** "
            f"{data_internacao.strftime('%d/%m/%Y')}"
        )

        st.write(
            f"**Data da cirurgia:** "
            f"{data_cirurgia.strftime('%d/%m/%Y')}"
        )

        st.write(
            f"**Tempo entre internação e cirurgia:** "
            f"{tempo_int_cir_dias} dia"
            f"{'s' if tempo_int_cir_dias != 1 else ''}"
        )

        st.write(
            "**ID da predição:**"
        )

        st.code(
            id_predicao
        )


# =========================================================
# ABA 2 - AUDITORIA
# =========================================================

with aba_auditoria:

    st.subheader(
        "Registrar desfecho real"
    )

    st.write(
        "Após a alta hospitalar, informe o prontuário "
        "para localizar a predição e registrar "
        "a data da alta."
    )


    # -----------------------------------------------------
    # BUSCA PELO PRONTUÁRIO
    # -----------------------------------------------------

    prontuario_busca = st.text_input(
        "Prontuário",
        key="auditoria_prontuario",
        placeholder="Digite o prontuário",
    )


    buscar = st.button(
        "Buscar predição",
        key="buscar_predicao",
    )


    if buscar:

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


    # -----------------------------------------------------
    # REGISTRO LOCALIZADO
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


        st.write(
            f"**Prontuário:** "
            f"{registro_auditoria.get('prontuario', '')}"
        )


        data_cirurgia_registro = None


        if registro_auditoria.get(
            "data_internacao"
        ):

            data_int = pd.to_datetime(
                registro_auditoria[
                    "data_internacao"
                ]
            )

            st.write(
                f"**Data da internação:** "
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
                ).date()
            )

            st.write(
                f"**Data da cirurgia:** "
                f"{data_cirurgia_registro.strftime('%d/%m/%Y')}"
            )


        if registro_auditoria.get(
            "tempo_int_cir_dias"
        ) is not None:

            st.write(
                f"**Tempo entre internação e cirurgia:** "
                f"{registro_auditoria['tempo_int_cir_dias']} dias"
            )


        # -------------------------------------------------
        # RESULTADO PREVISTO
        # -------------------------------------------------

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
        # CASO JÁ AUDITADO
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


            if registro_auditoria.get(
                "data_alta"
            ):

                data_alta_registro = (
                    pd.to_datetime(
                        registro_auditoria[
                            "data_alta"
                        ]
                    ).date()
                )

                st.write(
                    f"**Data da alta:** "
                    f"{data_alta_registro.strftime('%d/%m/%Y')}"
                )


            if registro_auditoria.get(
                "dias_reais_internacao"
            ) is not None:

                st.write(
                    f"**Dias reais após a cirurgia:** "
                    f"{registro_auditoria['dias_reais_internacao']} dias"
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
        # REGISTRAR DATA DA ALTA
        # -------------------------------------------------

        else:

            if data_cirurgia_registro is None:

                st.error(
                    "Este registro não possui data da cirurgia."
                )

                st.stop()


            st.markdown(
                "### Informar alta hospitalar"
            )


            data_alta = st.date_input(
                "Data da alta",
                value=data_cirurgia_registro,
                min_value=data_cirurgia_registro,
                format="DD/MM/YYYY",
                key="auditoria_data_alta",
            )


            # -------------------------------------------------
            # CÁLCULO AUTOMÁTICO DOS DIAS REAIS
            # -------------------------------------------------

            dias_reais = (
                data_alta
                - data_cirurgia_registro
            ).days


            st.info(
                f"**Dias reais de internação após a cirurgia:** "
                f"{dias_reais} dia"
                f"{'s' if dias_reais != 1 else ''}."
            )


            if dias_reais > 7:

                st.write(
                    "**Desfecho calculado:** "
                    "Internação prolongada"
                )

            else:

                st.write(
                    "**Desfecho calculado:** "
                    "Internação não prolongada"
                )


            salvar_desfecho = st.button(
                "Registrar alta e auditar modelo",
                type="primary",
            )


            if salvar_desfecho:

                if data_alta < data_cirurgia_registro:

                    st.error(
                        "A data da alta não pode ser "
                        "anterior à data da cirurgia."
                    )

                    st.stop()


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


                    st.success(
                        "Alta e desfecho registrados "
                        "com sucesso."
                    )


                    st.markdown(
                        "### Resultado da auditoria"
                    )


                    st.write(
                        f"**Data da alta:** "
                        f"{data_alta.strftime('%d/%m/%Y')}"
                    )

                    st.write(
                        f"**Dias reais após a cirurgia:** "
                        f"{dias_reais} dia"
                        f"{'s' if dias_reais != 1 else ''}"
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
        "O tempo entre internação e cirurgia é "
        "calculado automaticamente a partir das datas."
    )

    st.write(
        "Na auditoria, a data da alta é informada "
        "e os dias reais após a cirurgia são calculados "
        "automaticamente."
    )

    st.write(
        "Internação prolongada é definida como "
        "mais de 7 dias entre a cirurgia e a alta."
    )

    st.markdown(
        """
**Regras de auditoria**

- **VP:** previu alto risco e houve internação prolongada.
- **VN:** previu baixo risco e não houve internação prolongada.
- **FP:** previu alto risco e não houve internação prolongada.
- **FN:** previu baixo risco e houve internação prolongada.

**VP e VN → Modelo acertou**

**FP e FN → Modelo errou**
        """
    )
