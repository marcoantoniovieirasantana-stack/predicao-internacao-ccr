"""Acesso e registro das simulações dos avaliadores IVC."""

VERSAO_APP = 'ivc2'
EMAILS_SEM_IVC = frozenset({
    'caugustomsousa@gmail.com',
    'marcoantonio.vieirasantana@yahoo.com.br',
})


def inscrito(supabase, email):
    email = email.strip().lower()
    if email in EMAILS_SEM_IVC:
        return False
    resposta = (supabase.table('ivc_avaliadores').select('ativo')
                .eq('email', email).limit(1).execute())
    return bool(resposta.data and resposta.data[0]['ativo'])


def registrar_teste(supabase, user_id, email, teste_id, probabilidade,
                    classificacao, registro):
    """Guarda a simulação fora da auditoria, sem prontuário ou datas exatas."""
    dados = {chave: valor for chave, valor in registro.items()
             if chave not in {
                 'id_predicao', 'prontuario', 'data_internacao', 'data_cirurgia',
                 'probabilidade', 'threshold', 'classificacao_prevista',
                 'modelo', 'usuario_criacao_id', 'usuario_criacao_email',
             }}
    supabase.table('ivc_testes').insert({
        'id': teste_id,
        'avaliador_id': user_id,
        'avaliador_email': email.strip().lower(),
        'codigo_caso': 'CASO_LIVRE',
        'versao_app': VERSAO_APP,
        'probabilidade': float(probabilidade),
        'classificacao': classificacao,
        'dados_entrada': dados,
        'threshold': float(registro['threshold']),
        'modelo': registro['modelo'],
    }).execute()
