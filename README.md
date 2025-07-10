# Chatbot com Flask e Google Gemini

Este projeto implementa um chatbot inteligente usando Flask e a API do Google Gemini. O chatbot pode ser facilmente adaptado para diferentes temas e conteúdos.

## Pré-requisitos

- Python 3.9 ou superior
- Conta Google Cloud com API Gemini ativada
- Chave de API do Google Gemini

## Instalação

1. Abra o powershel ou cmd e clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd TEI_4301_2024
   ```
   Você também pode fazer o download se preferir.

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure o arquivo `.env`:
   ```plaintext
   GOOGLE_API_KEY="sua_chave_api_aqui"
   ```

## Adaptando para Seu Tema

1. **Edite o Conteúdo Base**
   - Abra `/app/utils/knowledge_base.py`
   - Substitua o texto existente com seu conteúdo
   - Mantenha a formatação markdown para melhor estrutura
   ```python
   CONHECIMENTO_BASE = """
   # Seu Tema

   ## Seção 1
   Conteúdo detalhado da seção 1...

   ## Seção 2
   Conteúdo detalhado da seção 2...
   """
   ```

2. **Ajuste o Prompt do Chatbot**
   - Em `app.py`, localize o prompt inicial
   - Modifique as instruções para seu contexto
   - Mantenha as regras de formatação e comportamento

3. **Personalize a Interface**
   - Cores e estilos em `static/style.css`
   - Textos e elementos em `templates/index.html`
   - Comportamentos em `static/script.js`

## Executando o Projeto

1. Inicie o servidor:
   ```bash
   python app.py
   ```

2. Acesse no navegador:
   ```
   http://localhost:5000
   ```

## Dicas para Melhor Uso

1. **Conteúdo Estruturado**
   - Use títulos e subtítulos claros
   - Organize informações em tópicos
   - Inclua exemplos relevantes

2. **Prompts Eficientes**
   - Seja específico nas instruções
   - Defina o tom das respostas
   - Estabeleça limites claros

3. **Manutenção**
   - Atualize o conteúdo regularmente
   - Monitore as respostas
   - Ajuste conforme necessário

## Solução de Problemas

- **API não responde**: Verifique a chave API no `.env`
- **Formatação incorreta**: Revise a sintaxe markdown
- **Respostas inadequadas**: Ajuste o prompt inicial

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar problemas
- Sugerir melhorias
- Enviar pull requests
