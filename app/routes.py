from flask import render_template, request, jsonify
import os
import google.generativeai as genai
from .utils.embedding_manager import EmbeddingManager

# Configuração do PaLM
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Instância global do EmbeddingManager
embedding_manager = EmbeddingManager(os.getenv('GOOGLE_API_KEY'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Busca o contexto relevante usando embeddings
        context = embedding_manager.search_query(user_message)
        
        # Modifica o prompt para incluir o contexto encontrado
        prompt = f"""Com base apenas no seguinte contexto: 
        {context}
        
        Responda à pergunta: {user_message}
        
        Responda de forma concisa e direta, em português."""
        
        # Gera a resposta usando o modelo
        response = model.generate_content(prompt)
        
        return jsonify({
            'response': response.text,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Erro: {str(e)}")
        return jsonify({
            'response': 'Desculpe, ocorreu um erro ao processar sua mensagem.',
            'status': 'error'
        }), 500
