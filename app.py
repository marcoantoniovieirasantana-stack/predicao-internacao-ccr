import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
# CONTROLE DE VERSÃO DO FORMULÁRIO
# =========================================================

if "form_version" not in st.session_state:
    st.session_state["form_version"] = 0


form_version = st.session_state["form_version"]


def iniciar_novo_paciente():

    # -----------------------------------------------------
    # Remove estados antigos dos campos clínicos
    # -----------------------------------------------------

    prefixos_clinicos = (
        "pred_",
        "prontuario_",
        "data_internacao_",
        "data_cirurgia_",
    )

    chaves_remover = [
        chave
        for chave in list(st.session_state.keys())
        if chave.startswith(prefixos_clinicos)
    ]

    for chave in chaves_remover:
        st.session_state.pop(
            chave,
            None,
        )

    # -----------------------------------------------------
    # Remove a última predição e explicação SHAP
    # -----------------------------------------------------

    st.session_state.pop(
        "ultima_predicao",
        None,
    )

    # -----------------------------------------------------
    # Cria uma nova versão do formulário
    # -----------------------------------------------------

    st.session_state["form_version"] = (
        st.session_state.get(
            "form_version",
            0,
        )
        + 1
    )


def sair_area_admin():

    st.session_state.pop(
        "admin_autenticado",
        None,
    )

    st.session_state.pop(
        "pagina_admin",
        None,
    )

    st.session_state.pop(
        "registro_auditoria",
        None,
    )

    st.session_state.pop(
        "dados_desempenho",
        None,
    )

    st.session_state.pop(
        "dados_csv",
        None,
    )

    st.rerun()


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


if threshold is None:

    st.error(
        "O bundle não possui ponto de corte definido."
    )

    st.stop()


# =========================================================
# SUPABASE
# =========================================================

def conectar_supabase():

    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]

    if "supabase_client" not in st.session_state:

        st.session_state["supabase_client"] = (
            create_client(
                url,
                key,
            )
        )

    return st.session_state["supabase_client"]


try:

    supabase = conectar_supabase()

except Exception as exc:

    st.error(
        "Não foi possível conectar ao banco de auditoria."
    )

    st.exception(
        exc
    )

    st.stop()


# =========================================================
# AUTENTICAÇÃO E PERFIL DE ACESSO
# =========================================================

def encerrar_sessao():

    try:

        supabase.auth.sign_out()

    except Exception:

        pass


    chaves_preservadas = {
        "form_version",
    }

    for chave in list(
        st.session_state.keys()
    ):

        if chave not in chaves_preservadas:

            st.session_state.pop(
                chave,
                None,
            )


    st.rerun()


def carregar_perfil_usuario(usuario):

    resposta_perfil = (
        supabase
        .table(
            "perfis_usuarios"
        )
        .select(
            "user_id,email,nome,perfil,ativo"
        )
        .eq(
            "user_id",
            str(usuario.id),
        )
        .limit(
            1
        )
        .execute()
    )


    registros_perfil = (
        resposta_perfil.data
        if resposta_perfil.data
        else []
    )


    if not registros_perfil:

        return None


    return registros_perfil[0]


usuario_atual = st.session_state.get(
    "usuario_atual"
)

perfil_usuario = st.session_state.get(
    "perfil_usuario"
)


if not usuario_atual or not perfil_usuario:

    st.title(
        "🏥 Predição de Internação Prolongada — CCR"
    )

    st.subheader(
        "Acesso à aplicação"
    )

    st.caption(
        "Utilize o e-mail previamente autorizado "
        "pelo administrador."
    )


    with st.form(
        "formulario_login",
        clear_on_submit=False,
    ):

        email_login = st.text_input(
            "E-mail",
            placeholder="nome@exemplo.com",
        )

        senha_login = st.text_input(
            "Senha",
            type="password",
        )

        entrar = st.form_submit_button(
            "Entrar",
            type="primary",
            use_container_width=True,
        )


    if entrar:

        if (
            not email_login.strip()
            or
            not senha_login
        ):

            st.warning(
                "Informe o e-mail e a senha."
            )


        else:

            try:

                resposta_login = (
                    supabase
                    .auth
                    .sign_in_with_password(
                        {
                            "email": email_login.strip().lower(),
                            "password": senha_login,
                        }
                    )
                )


                usuario_login = resposta_login.user


                if usuario_login is None:

                    raise ValueError(
                        "Usuário não localizado."
                    )


                perfil_login = carregar_perfil_usuario(
                    usuario_login
                )


                if perfil_login is None:

                    supabase.auth.sign_out()

                    st.error(
                        "Este usuário não está autorizado "
                        "a acessar a aplicação."
                    )


                elif not perfil_login.get(
                    "ativo",
                    False,
                ):

                    supabase.auth.sign_out()

                    st.error(
                        "Este usuário está bloqueado. "
                        "Procure o administrador."
                    )


                elif perfil_login.get(
                    "perfil"
                ) not in {
                    "digitador",
                    "auditor",
                    "administrador",
                }:

                    supabase.auth.sign_out()

                    st.error(
                        "O perfil deste usuário é inválido."
                    )


                else:

                    st.session_state[
                        "usuario_atual"
                    ] = {
                        "id": str(usuario_login.id),
                        "email": usuario_login.email,
                    }

                    st.session_state[
                        "perfil_usuario"
                    ] = perfil_login

                    st.rerun()


            except Exception:

                st.error(
                    "E-mail ou senha inválidos."
                )


    st.stop()


usuario_id = str(
    usuario_atual.get(
        "id",
        "",
    )
)

usuario_email = str(
    usuario_atual.get(
        "email",
        "",
    )
)

nome_usuario = (
    perfil_usuario.get(
        "nome"
    )
    or
    usuario_email
)

tipo_perfil = perfil_usuario.get(
    "perfil",
    "digitador",
)


def registrar_log_operacao(
    operacao,
    id_registro=None,
    detalhes=None,
):

    try:

        registro_log = {
            "usuario_id": usuario_id,
            "usuario_email": usuario_email,
            "operacao": operacao,
            "id_registro": (
                str(id_registro)
                if id_registro is not None
                else None
            ),
            "detalhes": detalhes or {},
        }

        (
            supabase
            .table(
                "log_operacoes"
            )
            .insert(
                registro_log
            )
            .execute()
        )


    except Exception:

        st.warning(
            "A operação foi concluída, mas não foi possível "
            "registrá-la no histórico de acessos."
        )


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

    if meta.get(
        "type"
    ) == "categorical":

        return formatar_categoria(
            valor
        )

    if isinstance(
        valor,
        float,
    ):

        if valor.is_integer():

            return str(
                int(
                    valor
                )
            )

    return str(
        valor
    )


# =========================================================
# CAMPOS
# =========================================================

valores = {}


def criar_campo(
    feature,
    versao_formulario,
):

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
    # ESTE CAMPO É CALCULADO PELAS DATAS
    # -----------------------------------------------------

    if feature == "tempo_int_cir_dias":
        return


    # -----------------------------------------------------
    # NUMÉRICO
    # -----------------------------------------------------

    if meta.get(
        "type"
    ) == "numeric":

        min_v = meta.get(
            "min"
        )

        max_v = meta.get(
            "max"
        )

        help_txt = None

        if (
            min_v is not None
            and
            max_v is not None
        ):

            help_txt = (
                "Faixa observada no banco de desenvolvimento: "
                f"{int(round(min_v))} a "
                f"{int(round(max_v))}."
            )

        valores[
            feature
        ] = st.number_input(
            label,
            min_value=(
                int(
                    round(
                        min_v
                    )
                )
                if min_v is not None
                else None
            ),
            max_value=(
                int(
                    round(
                        max_v
                    )
                )
                if max_v is not None
                else None
            ),
            value=None,
            step=1,
            format="%d",
            placeholder="Informe o valor",
            help=help_txt,
            key=(
                f"pred_"
                f"{feature}_"
                f"{versao_formulario}"
            ),
        )


    # -----------------------------------------------------
    # CATEGÓRICO
    # -----------------------------------------------------

    elif meta.get(
        "type"
    ) == "categorical":

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
                placeholder="Selecione uma opção",
                format_func=formatar_categoria,
                key=(
                    f"pred_"
                    f"{feature}_"
                    f"{versao_formulario}"
                ),
            )

        else:

            valores[
                feature
            ] = st.text_input(
                label,
                value="",
                key=(
                    f"pred_"
                    f"{feature}_"
                    f"{versao_formulario}"
                ),
            )


    # -----------------------------------------------------
    # OUTROS
    # -----------------------------------------------------

    else:

        valores[
            feature
        ] = st.text_input(
            label,
            value="",
            key=(
                f"pred_"
                f"{feature}_"
                f"{versao_formulario}"
            ),
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

    return (
        f"{valor:.1%}"
    )


# =========================================================
# SHAP
# =========================================================

def identificar_variavel_original(
    feature_transformada
):

    nome = str(
        feature_transformada
    )

    nome = (
        nome
        .replace(
            "num__",
            "",
        )
        .replace(
            "cat__",
            "",
        )
        .replace(
            "nom__",
            "",
        )
    )


    for original in sorted(
        predictors,
        key=len,
        reverse=True,
    ):

        if nome == original:

            return original

        if nome.startswith(
            original
            + "_"
        ):

            return original


    return nome


def calcular_shap_individual(
    novo_paciente
):

    # Importação apenas quando necessário
    import shap

    pipeline = model


    if not hasattr(
        pipeline,
        "named_steps",
    ):

        raise ValueError(
            "O modelo salvo não possui "
            "pipeline compatível com SHAP."
        )


    preprocessador = (
        pipeline
        .named_steps[
            "preprocess"
        ]
    )


    modelo_shap = (
        pipeline
        .named_steps[
            "model"
        ]
    )


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

        str(
            nome
        )
        .replace(
            "num__",
            "",
        )
        .replace(
            "cat__",
            "",
        )
        .replace(
            "nom__",
            "",
        )

        for nome
        in nomes_features
    ]


    X_proc_df = pd.DataFrame(
        X_proc,
        columns=nomes_features_limpos,
    )


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
            valores_shap[
                0
            ]
        )


    else:

        valores_shap = (
            valores_shap
            .reshape(
                -1
            )
        )


    tabela_transformada = pd.DataFrame(
        {
            "feature_transformada":
                nomes_features_limpos,

            "valor_shap":
                valores_shap,
        }
    )


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
                    str(
                        x
                    )
                    .replace(
                        "_",
                        " ",
                    )
                    .capitalize(),
                )
        )
    )


    valores_originais = (
        novo_paciente
        .iloc[
            0
        ]
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


    return (
        tabela_agregada
        .sort_values(
            "impacto_absoluto",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# CONSULTA COMPLETA DO SUPABASE
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
            .select(
                "*"
            )
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


        if len(
            lote
        ) < tamanho_lote:

            break


        inicio += tamanho_lote


    return todos


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
# USUÁRIO E ÁREAS RESTRITAS
# =========================================================

st.sidebar.markdown(
    "## 👤 Usuário"
)

st.sidebar.write(
    f"**{nome_usuario}**"
)

st.sidebar.caption(
    usuario_email
)

rotulos_perfil = {
    "digitador": "Digitador",
    "auditor": "Auditor",
    "administrador": "Administrador",
}

st.sidebar.write(
    "**Perfil:** "
    f"{rotulos_perfil.get(tipo_perfil, tipo_perfil)}"
)


if st.sidebar.button(
    "🚪 Sair da aplicação",
    use_container_width=True,
):

    encerrar_sessao()


st.sidebar.divider()


admin_autenticado = tipo_perfil in {
    "auditor",
    "administrador",
}


pagina_admin = None


if admin_autenticado:

    st.sidebar.markdown(
        "## 🔐 Área restrita"
    )


    opcoes_area = [
        "—",
        "📋 Auditoria",
    ]


    if tipo_perfil == "administrador":

        opcoes_area.extend(
            [
                "📊 Desempenho do modelo",
                "📥 Gerar planilha CSV",
            ]
        )


    pagina_admin = (
        st.sidebar
        .radio(
            "Área restrita",
            opcoes_area,
            label_visibility="collapsed",
            key="pagina_admin",
        )
    )


    if st.sidebar.button(
        "↩️ Voltar à predição",
        use_container_width=True,
    ):

        sair_area_admin()


else:

    st.sidebar.info(
        "Perfil autorizado para registrar novas predições."
    )


st.sidebar.divider()


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
    f"**Ponto de corte:** "
    f"{threshold:.0%}"
)


if sensibilidade_meta is not None:

    st.sidebar.write(
        f"**Meta de sensibilidade:** "
        f"{sensibilidade_meta:.0%}"
    )


# =========================================================
# DEFINE MODO
# =========================================================

modo_admin = (
    admin_autenticado
    and
    pagina_admin
    not in [
        None,
        "—",
    ]
)


# =========================================================
# ÁREA CLÍNICA
# =========================================================

if not modo_admin:


    aba_predicao, aba_explicacao = (
        st.tabs(
            [
                "🔎 Nova predição",
                "🧠 Entenda a decisão",
            ]
        )
    )


    # =====================================================
    # ABA — NOVA PREDIÇÃO
    # =====================================================

    with aba_predicao:


        col_titulo, col_novo = (
            st.columns(
                [
                    4,
                    1,
                ]
            )
        )


        with col_titulo:

            st.markdown(
                "### Cadastro para predição"
            )


        with col_novo:

            st.button(
                "➕ Iniciar novo paciente",
                use_container_width=True,
                on_click=iniciar_novo_paciente,
            )


        # Atualiza a versão após eventual callback
        form_version = (
            st.session_state[
                "form_version"
            ]
        )


        # =================================================
        # IDENTIFICAÇÃO
        # =================================================

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🪪 Identificação do paciente"
            )


            col1, col2, col3 = (
                st.columns(
                    3
                )
            )


            with col1:

                prontuario = st.text_input(
                    "Prontuário",
                    value="",
                    placeholder="Digite o prontuário",
                    key=(
                        f"prontuario_"
                        f"{form_version}"
                    ),
                )


            with col2:

                data_internacao = (
                    st.date_input(
                        "Data da internação",
                        value=None,
                        format="DD/MM/YYYY",
                        key=(
                            f"data_internacao_"
                            f"{form_version}"
                        ),
                    )
                )


            with col3:

                data_cirurgia = (
                    st.date_input(
                        "Data da cirurgia",
                        value=None,
                        format="DD/MM/YYYY",
                        key=(
                            f"data_cirurgia_"
                            f"{form_version}"
                        ),
                    )
                )


            # =============================================
            # INTERVALO INTERNAÇÃO → CIRURGIA
            # =============================================

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


        # =================================================
        # DADOS PARA PREDIÇÃO
        # =================================================

        st.markdown(
            "## Dados para predição"
        )


        linha1_col1, linha1_col2 = (
            st.columns(
                2
            )
        )


        # -------------------------------------------------
        # DADOS DO PACIENTE
        # -------------------------------------------------

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
                            feature,
                            form_version,
                        )


        # -------------------------------------------------
        # DADOS ONCOLÓGICOS
        # -------------------------------------------------

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
                            feature,
                            form_version,
                        )


        linha2_col1, linha2_col2 = (
            st.columns(
                2
            )
        )


        # -------------------------------------------------
        # PROCEDIMENTO CIRÚRGICO
        # -------------------------------------------------

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
                            feature,
                            form_version,
                        )


        # -------------------------------------------------
        # INTERNAÇÃO E CUIDADOS
        # -------------------------------------------------

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
                            feature,
                            form_version,
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
                        "e cirurgia será calculado automaticamente."
                    )


        # =================================================
        # OUTRAS VARIÁVEIS
        # =================================================

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
                        feature,
                        form_version,
                    )


        # =================================================
        # BOTÃO CALCULAR
        # =================================================

        calcular = st.button(
            "🧠 Calcular risco de internação prolongada",
            type="primary",
            use_container_width=True,
            key=(
                f"calcular_"
                f"{form_version}"
            ),
        )


        # =================================================
        # EXECUTAR PREDIÇÃO
        # =================================================

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


            if tempo_int_cir_dias is None:

                st.error(
                    "Não foi possível calcular o intervalo "
                    "entre internação e cirurgia."
                )

                st.stop()


            valores[
                "tempo_int_cir_dias"
            ] = int(
                tempo_int_cir_dias
            )


            # =============================================
            # VALIDAR CAMPOS
            # =============================================

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


                st.write(
                    "**Campos pendentes:**"
                )


                for campo in campos_faltantes:

                    st.write(
                        f"- {campo}"
                    )


                st.stop()


            # =============================================
            # DATAFRAME
            # =============================================

            novo_paciente = (
                pd.DataFrame(
                    [
                        valores
                    ],
                    columns=predictors,
                )
            )


            # =============================================
            # PROBABILIDADE
            # =============================================

            try:

                prob = float(
                    model
                    .predict_proba(
                        novo_paciente
                    )[
                        0,
                        1
                    ]
                )


            except Exception as exc:

                st.error(
                    "Não foi possível calcular a predição."
                )

                st.exception(
                    exc
                )

                st.stop()


            # =============================================
            # CLASSIFICAÇÃO
            # =============================================

            classificacao_prevista = (
                "Alto risco"
                if prob >= threshold
                else "Baixo risco"
            )


            id_predicao = str(
                uuid4()
            )


            # =============================================
            # REGISTRO
            # =============================================

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

                "usuario_criacao_id":
                    usuario_id,

                "usuario_criacao_email":
                    usuario_email,
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
                    meta.get(
                        "type"
                    )
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


            # =============================================
            # SALVAR NO SUPABASE
            # =============================================

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

                registrar_log_operacao(
                    "criacao_predicao",
                    id_predicao,
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


            # =============================================
            # GUARDAR PARA SHAP
            # =============================================

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
                    novo_paciente
                    .to_dict(
                        orient="records"
                    )[
                        0
                    ],
            }


            # =============================================
            # RESULTADO
            # =============================================

            st.markdown(
                "## Resultado da predição"
            )


            st.metric(
                "Probabilidade estimada de internação > 7 dias",
                f"{prob:.1%}",
            )


            if (
                classificacao_prevista
                == "Alto risco"
            ):

                st.error(
                    "🔴 ALTO RISCO — "
                    f"probabilidade {prob:.1%} ≥ "
                    f"ponto de corte {threshold:.0%}."
                )


            else:

                st.success(
                    "🟢 BAIXO RISCO — "
                    f"probabilidade {prob:.1%} < "
                    f"ponto de corte {threshold:.0%}."
                )


            st.success(
                "Predição registrada no banco de auditoria."
            )


            st.info(
                "Abra **🧠 Entenda a decisão** "
                "para visualizar a explicação individual."
            )


    # =====================================================
    # ABA — ENTENDA A DECISÃO
    # =====================================================

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


            col1, col2, col3 = (
                st.columns(
                    3
                )
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
                    "Ponto de corte",
                    f"{threshold:.0%}",
                )


            # =============================================
            # SHAP
            # =============================================

            dados_explicacao = (
                ultima_predicao[
                    "dados"
                ]
            )


            paciente_explicacao = (
                pd.DataFrame(
                    [
                        dados_explicacao
                    ],
                    columns=predictors,
                )
            )


            try:

                tabela_shap = (
                    calcular_shap_individual(
                        paciente_explicacao
                    )
                )


                top_shap = (
                    tabela_shap
                    .head(
                        10
                    )
                    .copy()
                )


                aumentam = (
                    top_shap[
                        top_shap[
                            "valor_shap"
                        ]
                        > 0
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
                        ]
                        < 0
                    ]
                    .sort_values(
                        "valor_shap",
                        ascending=True,
                    )
                )


                col_a, col_r = (
                    st.columns(
                        2
                    )
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


                # -----------------------------------------
                # GRÁFICO
                # -----------------------------------------

                import matplotlib.pyplot as plt


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
                    figsize=(
                        9,
                        6,
                    )
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
                    "Contribuição das variáveis clínicas"
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
                    "SHAP descreve como o modelo construiu "
                    "esta predição. Não representa causalidade."
                )


            except Exception as exc:

                st.warning(
                    "Não foi possível gerar a explicação SHAP."
                )


                with st.expander(
                    "Detalhes técnicos"
                ):

                    st.exception(
                        exc
                    )


            # =============================================
            # DCA
            # =============================================

            st.divider()


            st.markdown(
                "## 📈 Utilidade clínica da decisão"
            )


            st.info(
                "Na amostra de desenvolvimento, o modelo apresentou "
                "benefício clínico em uma faixa de pontos de corte "
                "entre **21% e 80%**."
            )


            st.write(
                f"O modelo utiliza **{threshold:.0%} como ponto de corte** "
                "para classificar o paciente como alto risco de internação "
                "prolongada. Esse valor está dentro da faixa em que o modelo "
                "apresentou benefício clínico."
            )


            st.caption(
                "Essa faixa representa utilidade clínica potencial "
                "dos diferentes pontos de corte e não significa que "
                "o modelo tenha maior precisão em todas as probabilidades "
                "entre 21% e 80%."
            )


# =========================================================
# ADMIN — AUDITORIA
# =========================================================

elif (
    admin_autenticado
    and
    pagina_admin
    == "📋 Auditoria"
):


    st.markdown(
        "## 📋 Auditoria"
    )


    st.caption(
        "Área administrativa restrita."
    )


    prontuario_busca = (
        st.text_input(
            "Prontuário",
            key="auditoria_prontuario",
            placeholder="Digite o prontuário",
        )
    )


    if st.button(
        "🔍 Buscar predição"
    ):

        st.session_state.pop(
            "registro_auditoria",
            None,
        )

        st.session_state.pop(
            "registros_auditoria",
            None,
        )

        st.session_state.pop(
            "auditoria_registro_selecionado",
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
                    .select(
                        "*"
                    )
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


                if registros:

                    st.session_state[
                        "registros_auditoria"
                    ] = registros


                else:

                    st.warning(
                        "Nenhuma predição encontrada."
                    )


            except Exception as exc:

                st.error(
                    "Erro ao consultar o banco de auditoria."
                )

                st.exception(
                    exc
                )


    if st.button(
        "🔎 Localizar registros sem prontuário",
        use_container_width=True,
        disabled=(
            tipo_perfil
            != "administrador"
        ),
    ):

        st.session_state.pop(
            "registro_auditoria",
            None,
        )

        st.session_state.pop(
            "registros_auditoria",
            None,
        )

        st.session_state.pop(
            "auditoria_registro_selecionado",
            None,
        )


        try:

            todos_registros = carregar_todos_registros()

            registros_sem_prontuario = [
                dict(registro)
                for registro in todos_registros
                if (
                    registro.get("prontuario") is None
                    or
                    not str(
                        registro.get("prontuario")
                    ).strip()
                )
            ]


            # Registros criados durante testes iniciais podem não
            # possuir todos os campos exigidos pela tela atual.
            # Os valores abaixo existem somente na memória da sessão
            # para permitir visualizar e excluir o registro; eles não
            # atualizam nem completam os dados armazenados no Supabase.
            for registro in registros_sem_prontuario:

                data_referencia = (
                    registro.get("data_predicao")
                    or
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

                registro["data_cirurgia"] = (
                    registro.get("data_cirurgia")
                    or
                    data_referencia
                )

                registro["probabilidade"] = (
                    registro.get("probabilidade")
                    if registro.get("probabilidade") is not None
                    else 0.0
                )

                registro["classificacao_prevista"] = (
                    registro.get("classificacao_prevista")
                    or
                    "Não informado"
                )

                registro["desfecho_real"] = (
                    registro.get("desfecho_real")
                    or
                    "Não informado — registro de teste"
                )

                registro["tipo_resultado"] = (
                    registro.get("tipo_resultado")
                    or
                    "Não informado"
                )

                registro["predicao"] = (
                    registro.get("predicao")
                    or
                    "Não informado"
                )


            if registros_sem_prontuario:

                st.session_state[
                    "registros_auditoria"
                ] = registros_sem_prontuario


            else:

                st.info(
                    "Nenhum registro sem prontuário foi encontrado."
                )


        except Exception as exc:

            st.error(
                "Erro ao localizar registros sem prontuário."
            )

            st.exception(
                exc
            )


    registros_auditoria = (
        st.session_state.get(
            "registros_auditoria",
            [],
        )
    )


    if registros_auditoria:

        st.success(
            f"{len(registros_auditoria)} registro(s) "
            "encontrado(s)."
        )


        def rotulo_registro_auditoria(indice):

            registro = registros_auditoria[indice]

            data_predicao = pd.to_datetime(
                registro.get("data_predicao"),
                errors="coerce",
            )

            data_texto = (
                data_predicao.strftime("%d/%m/%Y %H:%M")
                if not pd.isna(data_predicao)
                else "data não informada"
            )

            situacao = (
                "auditado"
                if registro.get("desfecho_real")
                else "aguardando alta"
            )

            return (
                f"{data_texto} | {situacao} | "
                f"ID: "
                f"{registro.get('id_predicao') or registro.get('id', '')}"
            )


        indice_selecionado = st.selectbox(
            "Selecione o lançamento",
            options=list(range(len(registros_auditoria))),
            format_func=rotulo_registro_auditoria,
            key="auditoria_registro_selecionado",
        )


        registro_auditoria = (
            registros_auditoria[indice_selecionado]
        )

        st.session_state[
            "registro_auditoria"
        ] = registro_auditoria


    else:

        registro_auditoria = None


    if registro_auditoria:


        data_cirurgia_registro = (
            pd.to_datetime(
                registro_auditoria[
                    "data_cirurgia"
                ]
            )
            .date()
        )


        st.write(
            f"**Prontuário:** "
            f"{registro_auditoria.get('prontuario') or 'Não informado'}"
        )


        st.write(
            f"**Cirurgia:** "
            f"{data_cirurgia_registro.strftime('%d/%m/%Y')}"
        )


        col1, col2 = (
            st.columns(
                2
            )
        )


        with col1:

            st.metric(
                "Probabilidade prevista",
                f"{float(registro_auditoria['probabilidade']):.1%}",
            )


        with col2:

            st.metric(
                "Classificação prevista",
                registro_auditoria[
                    "classificacao_prevista"
                ],
            )


        # -------------------------------------------------
        # JÁ AUDITADO
        # -------------------------------------------------

        if (
            registro_auditoria.get(
                "desfecho_real"
            )
        ):

            if (
                registro_auditoria.get(
                    "prontuario"
                ) is None
                or
                not str(
                    registro_auditoria.get(
                        "prontuario"
                    )
                ).strip()
            ):

                st.info(
                    "Registro antigo sem prontuário selecionado "
                    "para conferência ou exclusão."
                )


            else:

                st.info(
                    "Este caso já foi auditado."
                )


            col1, col2, col3 = (
                st.columns(
                    3
                )
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
                    "Resultado",
                    registro_auditoria[
                        "predicao"
                    ],
                )


        # -------------------------------------------------
        # REGISTRAR ALTA
        # -------------------------------------------------

        else:

            data_alta = (
                st.date_input(
                    "Data da alta",
                    value=None,
                    min_value=data_cirurgia_registro,
                    format="DD/MM/YYYY",
                    key="auditoria_data_alta",
                )
            )


            if data_alta is not None:

                dias_reais = (
                    data_alta
                    - data_cirurgia_registro
                ).days


                col1, col2 = (
                    st.columns(
                        2
                    )
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


                if st.button(
                    "✅ Registrar alta e auditar modelo",
                    type="primary",
                    use_container_width=True,
                ):

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

                        "usuario_auditoria_id":
                            usuario_id,

                        "usuario_auditoria_email":
                            usuario_email,
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

                        registrar_log_operacao(
                            "registro_alta_auditoria",
                            registro_auditoria[
                                "id_predicao"
                            ],
                        )


                        st.success(
                            "Auditoria registrada com sucesso."
                        )


                        st.session_state.pop(
                            "registro_auditoria",
                            None,
                        )

                        st.session_state.pop(
                            "registros_auditoria",
                            None,
                        )


                    except Exception as exc:

                        st.error(
                            "Não foi possível registrar a auditoria."
                        )

                        st.exception(
                            exc
                        )


        # -------------------------------------------------
        # EXCLUIR REGISTRO
        # -------------------------------------------------

        st.divider()


        with st.expander(
            "🗑️ Excluir registro"
        ):

            if tipo_perfil != "administrador":

                st.info(
                    "Somente o administrador pode excluir registros."
                )

            st.warning(
                "A exclusão é definitiva. Confira se este "
                "é realmente o lançamento que deseja excluir."
            )


            valor_prontuario_registro = (
                registro_auditoria.get(
                    "prontuario"
                )
            )

            prontuario_ausente = (
                valor_prontuario_registro is None
                or
                not str(
                    valor_prontuario_registro
                ).strip()
            )

            prontuario_registro = (
                ""
                if prontuario_ausente
                else str(
                    valor_prontuario_registro
                ).strip()
            )

            nome_campo_id = (
                "id_predicao"
                if registro_auditoria.get(
                    "id_predicao"
                )
                else "id"
            )

            valor_id_predicao = (
                registro_auditoria.get(
                    nome_campo_id
                )
            )

            id_predicao_registro = (
                ""
                if valor_id_predicao is None
                else str(
                    valor_id_predicao
                ).strip()
            )


            st.write(
                f"**Prontuário do registro:** "
                f"{prontuario_registro or 'Não informado'}"
            )

            st.code(
                id_predicao_registro,
                language=None,
            )

            st.caption(
                f"Identificador utilizado: {nome_campo_id}"
            )


            if prontuario_ausente:

                st.info(
                    "Como este registro não possui prontuário, "
                    "a confirmação será feita pelo ID único da predição."
                )

                prontuario_confirmacao = ""


            else:

                prontuario_confirmacao = st.text_input(
                    "Confirme o prontuário",
                    key=(
                        "excluir_prontuario_"
                        f"{id_predicao_registro}"
                    ),
                )

            id_confirmacao = st.text_input(
                "Confirme o ID exibido",
                key=(
                    "excluir_id_"
                    f"{id_predicao_registro}"
                ),
            )

            confirmar_exclusao = st.checkbox(
                "Confirmo que selecionei o registro correto.",
                key=(
                    "confirmar_exclusao_"
                    f"{id_predicao_registro}"
                ),
            )


            dados_conferem = (
                bool(id_predicao_registro)
                and
                id_confirmacao.strip()
                == id_predicao_registro
                and
                confirmar_exclusao
                and
                (
                    prontuario_ausente
                    or
                    prontuario_confirmacao.strip()
                    == prontuario_registro
                )
            )


            if st.button(
                "Excluir registro definitivamente",
                type="primary",
                use_container_width=True,
                disabled=(
                    not dados_conferem
                    or
                    tipo_perfil
                    != "administrador"
                ),
                key=(
                    "botao_excluir_"
                    f"{id_predicao_registro}"
                ),
            ):

                try:

                    consulta_exclusao = (
                        supabase
                        .table(
                            "auditoria_predicoes"
                        )
                        .delete()
                        .eq(
                            nome_campo_id,
                            id_predicao_registro,
                        )
                    )


                    if not prontuario_ausente:

                        consulta_exclusao = (
                            consulta_exclusao
                            .eq(
                                "prontuario",
                                prontuario_registro,
                            )
                        )


                    resposta_exclusao = (
                        consulta_exclusao.execute()
                    )


                    if resposta_exclusao.data:

                        registrar_log_operacao(
                            "exclusao_registro",
                            id_predicao_registro,
                            {
                                "campo_identificador":
                                    nome_campo_id,
                            },
                        )

                        st.session_state.pop(
                            "registro_auditoria",
                            None,
                        )

                        st.session_state.pop(
                            "registros_auditoria",
                            None,
                        )

                        st.success(
                            "Registro excluído com sucesso."
                        )

                        st.rerun()


                    else:

                        st.error(
                            "Nenhum registro foi excluído. "
                            "Atualize a busca e confira o prontuário e o ID."
                        )


                except Exception as exc:

                    st.error(
                        "Não foi possível excluir o registro."
                    )

                    st.exception(
                        exc
                    )


# =========================================================
# ADMIN — DESEMPENHO
# =========================================================

elif (
    admin_autenticado
    and
    tipo_perfil == "administrador"
    and
    pagina_admin
    == "📊 Desempenho do modelo"
):


    st.markdown(
        "## 📊 Desempenho do modelo"
    )


    st.caption(
        "Área administrativa restrita. "
        "As métricas utilizam somente casos auditados."
    )


    if st.button(
        "🔄 Carregar / atualizar indicadores"
    ):

        with st.spinner(
            "Consultando banco de auditoria..."
        ):

            try:

                registros = (
                    carregar_todos_registros()
                )


                st.session_state[
                    "dados_desempenho"
                ] = (
                    registros
                )


            except Exception as exc:

                st.error(
                    "Não foi possível consultar a base."
                )

                st.exception(
                    exc
                )


    registros = (
        st.session_state.get(
            "dados_desempenho"
        )
    )


    if not registros:

        st.info(
            "Clique em **Carregar / atualizar indicadores**."
        )


    else:

        df = pd.DataFrame(
            registros
        )


        if (
            "desfecho_real"
            in df.columns
        ):

            auditados = (
                df[
                    df[
                        "desfecho_real"
                    ]
                    .notna()
                    &
                    (
                        df[
                            "desfecho_real"
                        ]
                        .astype(
                            str
                        )
                        .str
                        .strip()
                        != ""
                    )
                ]
                .copy()
            )


        else:

            auditados = (
                pd.DataFrame()
            )


        total = len(
            df
        )

        total_auditados = len(
            auditados
        )


        col1, col2, col3 = (
            st.columns(
                3
            )
        )


        with col1:

            st.metric(
                "Total de predições",
                total,
            )


        with col2:

            st.metric(
                "Casos auditados",
                total_auditados,
            )


        with col3:

            st.metric(
                "Pendentes",
                total
                - total_auditados,
            )


        if auditados.empty:

            st.warning(
                "Ainda não existem casos suficientes "
                "com desfecho registrado."
            )


        else:

            tipos = (
                auditados[
                    "tipo_resultado"
                ]
                .fillna(
                    ""
                )
                .astype(
                    str
                )
                .str
                .upper()
                .str
                .strip()
            )


            vp = int(
                (
                    tipos
                    == "VP"
                )
                .sum()
            )


            vn = int(
                (
                    tipos
                    == "VN"
                )
                .sum()
            )


            fp = int(
                (
                    tipos
                    == "FP"
                )
                .sum()
            )


            fn = int(
                (
                    tipos
                    == "FN"
                )
                .sum()
            )


            # ---------------------------------------------
            # VP VN FP FN
            # ---------------------------------------------

            col1, col2, col3, col4 = (
                st.columns(
                    4
                )
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


            # ---------------------------------------------
            # MÉTRICAS
            # ---------------------------------------------

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
                "### Indicadores"
            )


            col1, col2, col3 = (
                st.columns(
                    3
                )
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
                st.columns(
                    2
                )
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


            # ---------------------------------------------
            # MATRIZ
            # ---------------------------------------------

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


# =========================================================
# ADMIN — CSV
# =========================================================

elif (
    admin_autenticado
    and
    tipo_perfil == "administrador"
    and
    pagina_admin
    == "📥 Gerar planilha CSV"
):


    st.markdown(
        "## 📥 Gerar planilha CSV"
    )


    st.caption(
        "Área administrativa restrita."
    )


    st.warning(
        "O arquivo contém prontuário e todos os dados "
        "registrados na aplicação."
    )


    if st.button(
        "🔄 Preparar base para exportação",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Consultando banco de dados..."
        ):

            try:

                registros_csv = (
                    carregar_todos_registros()
                )


                st.session_state[
                    "dados_csv"
                ] = registros_csv


            except Exception as exc:

                st.error(
                    "Não foi possível consultar a base."
                )

                st.exception(
                    exc
                )


    registros_csv = (
        st.session_state.get(
            "dados_csv"
        )
    )


    if not registros_csv:

        st.info(
            "Clique em **Preparar base para exportação**."
        )


    else:

        df_csv = pd.DataFrame(
            registros_csv
        )


        st.success(
            f"Base preparada: "
            f"{len(df_csv)} registros."
        )


        # -------------------------------------------------
        # ORDEM DAS COLUNAS
        # -------------------------------------------------

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
            for coluna
            in colunas_prioritarias
            if coluna
            in df_csv.columns
        ]


        outras_colunas = [
            coluna
            for coluna
            in df_csv.columns
            if coluna
            not in colunas_existentes
        ]


        df_exportacao = (
            df_csv[
                colunas_existentes
                + outras_colunas
            ]
            .copy()
        )


        # -------------------------------------------------
        # VISUALIZAR
        # -------------------------------------------------

        with st.expander(
            "Visualizar registros"
        ):

            st.dataframe(
                df_exportacao,
                use_container_width=True,
                hide_index=True,
            )


        # -------------------------------------------------
        # CSV
        # -------------------------------------------------

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


        st.download_button(
            label="📥 Baixar planilha CSV completa",
            data=csv_completo,
            file_name=(
                "auditoria_modelo_ccr_"
                f"{datetime.now().strftime('%Y%m%d')}"
                ".csv"
            ),
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
