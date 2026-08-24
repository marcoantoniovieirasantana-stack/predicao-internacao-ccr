import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="CCR | Risco de internação prolongada",
    page_icon="🏥",
    layout="centered",
)

BUNDLE_PATH = Path(__file__).with_name("27_deployment_bundle.joblib")

@st.cache_resource
def carregar_bundle():
    return joblib.load(BUNDLE_PATH)

bundle = carregar_bundle()
model = bundle["model"]
predictors = bundle["predictors"]
schema = bundle.get("ui_schema", {})
threshold = bundle.get("clinical_threshold")
family = bundle.get("family", "Modelo")

st.title("Predição de internação prolongada")
st.caption("Pacientes submetidos à cirurgia por câncer colorretal")

st.info(
    "Protótipo de apoio à decisão para estimar risco de internação > 7 dias. "
    "Não substitui avaliação clínica e requer validação externa/prospectiva antes de uso assistencial."
)

if threshold is None:
    st.error(
        "O bundle não possui threshold operacional definido. "
        "Retorne ao notebook e revise a etapa de seleção do threshold."
    )
    st.stop()

st.sidebar.header("Modelo")
st.sidebar.write(f"**Algoritmo:** {family}")
st.sidebar.write(f"**Threshold operacional:** {threshold:.2f}")
st.sidebar.write(
    f"**Meta de sensibilidade:** {bundle.get('min_sensitivity_target', float('nan')):.0%}"
)
st.sidebar.caption(bundle.get("threshold_rule", ""))

st.subheader("Dados do paciente")

valores = {}

for feature in predictors:
    meta = schema.get(feature, {})
    label = meta.get("label", feature)

    if meta.get("type") == "numeric":
        min_v = meta.get("min")
        max_v = meta.get("max")
        median_v = meta.get("median")

        # Streamlit aceita entrada numérica sem impor min/max.
        # Os limites observados no desenvolvimento são mostrados apenas como ajuda.
        help_txt = None
        if min_v is not None and max_v is not None:
            help_txt = f"Faixa observada no banco de desenvolvimento: {min_v:g} a {max_v:g}."

        default = 0 if median_v is None else int(round(median_v))
        valores[feature] = st.number_input(
            step=1,
            format="%d",
            label,
            value=default,
            help=help_txt,
            key=feature,
        )

    elif meta.get("type") == "categorical":
        options = meta.get("options", [])
        if not options:
            valores[feature] = st.text_input(label, key=feature)
        else:
            valores[feature] = st.selectbox(
                label,
                options=options,
                key=feature,
            )
    else:
        valores[feature] = st.text_input(label, key=feature)

st.divider()

if st.button("Calcular risco", type="primary", use_container_width=True):
    novo_paciente = pd.DataFrame([valores], columns=predictors)

    try:
        prob = float(model.predict_proba(novo_paciente)[0, 1])
    except Exception as exc:
        st.error("Não foi possível calcular a predição com os dados informados.")
        st.exception(exc)
        st.stop()

    alto_risco = prob >= threshold

    st.subheader("Resultado")
    st.metric("Probabilidade estimada de internação > 7 dias", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if alto_risco:
        st.error(
            f"ALTO RISCO — probabilidade {prob:.1%} ≥ threshold {threshold:.0%}."
        )
    else:
        st.success(
            f"BAIXO RISCO — probabilidade {prob:.1%} < threshold {threshold:.0%}."
        )

    st.caption(
        "A classificação decorre do threshold operacional definido no notebook. "
        "O valor de probabilidade deve ser interpretado no contexto do desempenho e da calibração do modelo."
    )

    with st.expander("Dados usados na predição"):
        exibicao = novo_paciente.T.reset_index()
        exibicao.columns = ["Variável", "Valor"]
        st.dataframe(exibicao, use_container_width=True, hide_index=True)

with st.expander("Sobre este protótipo"):
    st.write(
        "O modelo foi desenvolvido para o desfecho de internação prolongada (>7 dias) "
        "após cirurgia por câncer colorretal. O Streamlit apenas aplica o pipeline salvo; "
        "não realiza treinamento ou recalibração."
    )
