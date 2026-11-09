"""Shared role formatting; keeps the existing tokenizer/vocabulary unchanged."""
ROLES = {'system':'System', 'user':'User', 'assistant':'Assistant'}


def encode_messages(tok, messages):
    ids, labels = [], []
    for m in messages:
        if m.get('role') not in ROLES or not isinstance(m.get('content'),str):
            raise ValueError('Invalid message')
        header = tok.encode('\n'+ROLES[m['role']]+': ').ids
        content = tok.encode(m['content']).ids
        if m['role']=='assistant':
            content += [tok.token_to_id('<eos>')]
        ids += header + content
        labels += [-100]*len(header) + (content if m['role']=='assistant' else [-100]*len(content))
    return ids, labels


def chat_prompt(tok, messages, budget):
    if not messages or messages[-1].get('role')!='user':
        raise ValueError('Conversation must end with a user message')
    messages=list(messages)
    if any(m.get('role') != ('user' if i%2==0 else 'assistant') for i,m in enumerate(messages)):
        raise ValueError('Expected alternating user and assistant messages')
    suffix=tok.encode('\nAssistant: ').ids
    while True:
        ids,_=encode_messages(tok,messages)
        if len(ids)+len(suffix)<=budget:
            return ids+suffix
        if len(messages)<=1:
            raise ValueError('Your latest message is too long for this model. Shorten it or request fewer tokens.')
        messages=messages[2:]
