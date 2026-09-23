"""Fluxo de avaliação IVC, isolado dos registros clínicos."""
import streamlit as st
from datetime import date
import unicodedata

VERSAO_INSTRUMENTO = '1.0'
VERSAO_APP = 'ivc1'
CODIGO_CASO = 'CASO_IVC_01'
PRONTUARIO_CASO = '666666'
DATA_INTERNACAO = date(2026, 1, 3)
DATA_CIRURGIA = date(2026, 1, 5)
NOME_PROCEDIMENTO = 'Amputação abdominoperineal de reto'
VALORES_CASO = {
    'f_idade_anos_int': 60,
    'idade_anos_diag': 59,
    'f_asa': 2,
    'f_localizacao': 'reto_inferior',
    'f_estagio': 'III',
    'f_neoadjuvancia': 's',
    'f_abord_cirurgica': 'laparoscopica',
    'tempo_cir_min2': 345,
    'num_orgaos_envolvidos': 1,
    'uti': 's',
    'urgencia': 'n',
    'sexo_int': 'feminino',
}

def _normalizar(valor):
    texto = unicodedata.normalize('NFKD', str(valor)).encode('ascii', 'ignore').decode()
    return texto.casefold().replace(' ', '').replace('_', '')

def valor_inicial(feature, meta, formatar_categoria):
    """Resolve the expected model value from the case and deployed schema."""
    alvo = VALORES_CASO.get(feature)
    if alvo is None:
        return None
    if meta.get('type') == 'numeric':
        # A numeric encoding for sex is unknown; leave it for verification.
        return None if feature == 'sexo_int' else int(alvo) if isinstance(alvo, int) else None
    if meta.get('type') != 'categorical':
        return None
    aliases = {
        'sexo_int': ('feminino', 'f', 'female'),
        'f_neoadjuvancia': ('s', 'sim', 'yes'),
        'uti': ('s', 'sim', 'yes'),
        'urgencia': ('n', 'nao', 'no'),
        'f_estagio': ('iii', '3', 'estagioiii'),
        'f_localizacao': ('retoinferior',),
        'f_abord_cirurgica': ('laparoscopica',),
    }
    validos = {_normalizar(v) for v in aliases.get(feature, (alvo,))}
    encontrados = [v for v in meta.get('options', [])
                  if _normalizar(v) in validos or _normalizar(formatar_categoria(v)) in validos]
    return encontrados[0] if len(encontrados) == 1 else None

def exibir_caso():
    st.info('Preencha os campos abaixo com os dados deste caso. A simulação não será gravada na auditoria clínica.')
    with st.expander('📋 Dados do caso para preenchimento', expanded=True):
        st.markdown(
            f'**Procedimento:** {NOME_PROCEDIMENTO}  \n'
            f'**Prontuário de teste:** {PRONTUARIO_CASO}  \n'
            '**Internação:** 03/01/2026 · **Cirurgia:** 05/01/2026  \n'
            '**Sexo:** Feminino · **Idade na internação:** 60 anos · **Idade ao diagnóstico:** 59 anos  \n'
            '**ASA:** 2 · **Localização:** Reto inferior · **Estágio:** III  \n'
            '**Neoadjuvância:** Sim · **Abordagem:** Laparoscópica · **Tempo cirúrgico:** 345 minutos  \n'
            '**Órgãos envolvidos:** 1 · **UTI:** Sim · **Urgência:** Não'
        )

def divergencias_caso(prontuario, data_internacao, data_cirurgia,
                      valores, predictors, schema, formatar_categoria):
    """Check every clinical input before calculating or saving the test."""
    divergencias = []
    if prontuario.strip() != PRONTUARIO_CASO:
        divergencias.append('Prontuário de teste')
    if data_internacao != DATA_INTERNACAO:
        divergencias.append('Data da internação')
    if data_cirurgia != DATA_CIRURGIA:
        divergencias.append('Data da cirurgia')
    for feature in predictors:
        if feature == 'tempo_int_cir_dias':
            esperado = (DATA_CIRURGIA - DATA_INTERNACAO).days
        else:
            esperado = valor_inicial(feature, schema.get(feature, {}), formatar_categoria)
        if esperado is None:
            divergencias.append('Configuração do caso: ' + feature)
            continue
        recebido = valores.get(feature)
        if str(recebido) != str(esperado):
            divergencias.append(schema.get(feature, {}).get('label', feature))
    return divergencias
ITENS = [
 'A finalidade, a população prevista e o momento da predição estão claros.',
 'Os dados solicitados são pertinentes e compreensíveis no pós-operatório imediato.',
 'O desfecho internação superior a sete dias está definido claramente.',
 'Os limites da predição e seu papel de apoio à decisão estão claros.',
 'As instruções e os nomes dos campos permitem preenchimento sem ambiguidade.',
 'A sequência e as mensagens permitem concluir o registro e consultar o resultado.',
 'A probabilidade e a classificação são apresentadas de modo compreensível.',
 'A apresentação visual favorece a leitura das informações principais.',
 'O resultado pode apoiar o planejamento assistencial e de recursos.',
 'O fluxo é compatível com a rotina do pós-operatório imediato.',
]

def inscrito(supabase, email):
    resposta = (supabase.table('ivc_avaliadores').select('ativo')
                .eq('email', email.lower()).limit(1).execute())
    return bool(resposta.data and resposta.data[0]['ativo'])

def registrar_teste(supabase, user_id, email, teste_id, probabilidade, classificacao):
    supabase.table('ivc_testes').insert({
        'id': teste_id, 'avaliador_id': user_id, 'avaliador_email': email.lower(),
        'codigo_caso': CODIGO_CASO, 'versao_app': VERSAO_APP,
        'probabilidade': float(probabilidade), 'classificacao': classificacao,
    }).execute()

def questionario(supabase, user_id, teste_id):
    st.markdown('## Questionário de avaliação da aplicação')
    st.caption('Responda individualmente após consultar a predição. O aplicativo não preenche suas notas.')
    try:
        anterior = (supabase.table('ivc_respostas').select('id')
                    .eq('avaliador_id', user_id).limit(1).execute().data)
    except Exception:
        st.error('Não foi possível consultar o estado da avaliação.')
        return
    if anterior:
        st.info('Sua avaliação já foi enviada.')
        return
    with st.form('formulario_ivc'):
        st.caption('1 Inadequado · 2 Exige revisão · 3 Adequado · 4 Muito adequado · NA Não avaliado')
        notas = {}
        for n, item in enumerate(ITENS, 1):
            notas[str(n)] = st.radio(f'{n}. {item}', [1, 2, 3, 4, 'NA'],
                                     index=None, horizontal=True, key=f'ivc_{teste_id}_{n}')
        justificativas = st.text_area('Motivos para notas 1 ou 2 e itens não avaliados')
        ausencias = st.text_area('Informações essenciais ausentes ou ajustes prioritários')
        dificuldades = st.text_area('Dificuldades ou falhas observadas')
        enviar = st.form_submit_button('Enviar avaliação', type='primary')
    if not enviar:
        return
    if any(n is None for n in notas.values()):
        st.warning('Responda aos dez itens antes de enviar.')
        return
    if any(n in (1, 2, 'NA') for n in notas.values()) and not justificativas.strip():
        st.warning('Explique as notas 1 e 2 ou os itens não avaliados.')
        return
    try:
        supabase.table('ivc_respostas').insert({
            'avaliador_id': user_id, 'teste_id': teste_id,
            'versao_instrumento': VERSAO_INSTRUMENTO,
            'notas': notas, 'justificativas': justificativas.strip(),
            'ausencias': ausencias.strip(), 'dificuldades': dificuldades.strip(),
        }).execute()
        st.success('Avaliação registrada com sucesso.')
        st.rerun()
    except Exception:
        st.error('Não foi possível registrar a avaliação. Tente novamente.')
