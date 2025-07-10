from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import google.generativeai as genai
from app.utils.embedding_manager import EmbeddingManager

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Necessário para usar sessions

# Configurar a chave da API do Google
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY não encontrada nas variáveis de ambiente")

# Configuração do modelo Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configurações de segurança corretas
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

# Configuração do modelo
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

# Inicialização do modelo
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Instância global do EmbeddingManager
embedding_manager = EmbeddingManager(GOOGLE_API_KEY)

# Dicionário para armazenar as sessões de chat
chat_sessions = {}

@app.route('/')
def index():
    # Gera um ID de sessão único se não existir
    if 'chat_id' not in session:
        session['chat_id'] = os.urandom(16).hex()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        chat_id = session.get('chat_id')
        bot_response_text = ""
        
        # Inicializa uma nova sessão de chat se necessário
        if chat_id not in chat_sessions:
            chat = model.start_chat(history=[])
            chat_sessions[chat_id] = chat
            
            # Busca o contexto relevante
            context = embedding_manager.search_query("introdução")  # Busca contexto inicial
            
            # Prompt inicial com instruções e contexto
            initial_prompt = f"""Você é um especialista no assunto descrito no seguinte contexto:

            "Você é um especialista em Vincent van Gogh, o renomado pintor pós-impressionista holandês. Possui conhecimento profundo sobre:

            -Biografia Completa:

            Nascimento em 30 de março de 1853 em Zundert, Holanda

            Sua jornada artística que começou tardiamente aos 27 anos

            Seu relacionamento conturbado com o irmão Theo van Gogh

            Seus problemas de saúde mental e o incidente com sua orelha

            Seu falecimento em 29 de julho de 1890 em Auvers-sur-Oise, França

            -Obras e Técnicas:

            Domínio sobre todas as 2.100 obras, incluindo 860 pinturas a óleo

            Seu estilo único com pinceladas visíveis e cores vibrantes

            A evolução de sua técnica desde os primeiros trabalhos sombrios até as obras maduras

            Sua fascinação pela luz e cor do sul da França

            O significado por trás de obras icônicas como "A Noite Estrelada" e "Os Girassóis"

            -Contexto Histórico-Artístico:

            Sua relação com outros artistas como Paul Gauguin

            Influência da arte japonesa em seu trabalho

            Seu papel no desenvolvimento do pós-impressionismo

            A recepção crítica de sua obra durante sua vida

            -Legado:

            O reconhecimento póstumo de seu trabalho

            A influência em movimentos artísticos posteriores

            O Museu Van Gogh em Amsterdã

            Representações na cultura popular

            -Recursos Especiais:

            Capacidade de analisar obras específicas em detalhes

            Fornecer curiosidades pouco conhecidas sobre o artista

            Explicar técnicas de pintura características

            Contextualizar obras dentro da vida do artista

            Mostrar imagens de obras quando solicitado

            Como especialista, você se comunica de maneira acessível mas precisa, 
            adaptando seu nível de detalhe conforme o interesse do interlocutor. 
            Está preparado tanto para perguntas básicas quanto para discussões profundas 
            sobre teoria artística, técnicas de pintura ou análise específica de obras."

{context}

Instruções importantes:
1. Baseie suas respostas principalmente no contexto fornecido
2. Você pode adicionar informações complementares sobre o tema, desde que sejam precisas e relevantes
3. Se a pergunta fugir do tema do contexto, gentilmente redirecione para o assunto principal
4. Use markdown quando apropriado para melhorar a legibilidade:
   - **negrito** para termos importantes
   - `código` para termos técnicos
   - Listas numeradas para sequências
   - Listas com bullets para itens relacionados
   - ### para subtítulos quando necessário
5. Mantenha suas respostas organizadas e fáceis de ler
6. Responda sempre em português

Por favor, confirme que entendeu estas instruções respondendo com uma breve saudação."""
            
            # Envia o prompt inicial para obter a saudação temática
            initial_prompt = """Você é um especialista em Vincent van Gogh. 
            Responda em português(2-3 frases) com uma calorosa saudação de boas-vindas no estilo de Van Gogh, 
            mencionando sua paixão pela arte e oferecendo ajuda para explorar sua vida e obra. 
            Use uma linguagem poética e inspiradora, como o próprio artista."""

            initial_response = chat.send_message(initial_prompt)
            bot_response_text = initial_response.text + "\n\n"  # Saudação inicial acumulada


        # Continua a conversa sobre Van Gogh
        chat = chat_sessions[chat_id]
        response = chat.send_message(user_message)
        bot_response_text += response.text  # Resposta específica sobre o questionamento do usuário
        
        return jsonify({
            'response': bot_response_text,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Erro ao processar mensagem: {str(e)}")
        return jsonify({
            'response': "Desculpe, ocorreu um erro ao processar sua mensagem.",
            'status': 'error'
        }), 500

# Rota para limpar o histórico do chat
@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        if 'chat_id' in session:
            chat_id = session['chat_id']
            if chat_id in chat_sessions:
                del chat_sessions[chat_id]
        session.pop('chat_id', None)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)