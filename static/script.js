document.addEventListener('DOMContentLoaded', function() {
    const chatbotButton = document.getElementById('chatbot-button');
    const chatbotContainer = document.getElementById('chatbot-container');
    const closeChatbot = document.getElementById('close-chatbot');
    const userInput = document.getElementById('user-input');

    chatbotButton.addEventListener('click', function() {
        chatbotContainer.style.display = 'flex';
    });

    closeChatbot.addEventListener('click', function() {
        chatbotContainer.style.display = 'none';
    });

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

let isChatbotOpen = false;

function toggleChatbot() {
    const chatbotContainer = document.getElementById('chatbot-container');
    isChatbotOpen = !isChatbotOpen;
    
    if (isChatbotOpen) {
        chatbotContainer.style.display = 'flex';
    } else {
        chatbotContainer.style.display = 'none';
        clearChat(); // Limpa o chat ao fechar
    }
}

function formatMarkdownToHTML(text) {
    // Formatação mais completa de markdown
    return text
        // Headers
        .replace(/### (.*?)\n/g, '<h3>$1</h3>')
        .replace(/## (.*?)\n/g, '<h2>$1</h2>')
        .replace(/# (.*?)\n/g, '<h1>$1</h1>')
        
        // Listas
        .replace(/^\s*\d+\.\s+(.+)/gm, '<li>$1</li>')  // Lista numerada
        .replace(/^\s*[\-\*]\s+(.+)/gm, '<li>$1</li>') // Lista com bullets
        
        // Estilo de texto
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // Negrito
        .replace(/\*(.*?)\*/g, '<em>$1</em>')              // Itálico
        .replace(/`(.*?)`/g, '<code>$1</code>')            // Código inline
        
        // Links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        
        // Blocos de código
        .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
        
        // Parágrafos e quebras de linha
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
}

function appendMessage(message, isUser) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'user-message' : 'chatbot-message';
    
    if (isUser) {
        messageDiv.textContent = `Você: ${message}`;
    } else {
        // Adiciona wrapper para mensagem do chatbot
        const botLabel = document.createElement('div');
        botLabel.className = 'chatbot-label';
        botLabel.textContent = 'Chatbot:';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = formatMarkdownToHTML(message);
        
        messageDiv.appendChild(botLabel);
        messageDiv.appendChild(messageContent);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    
    if (message) {
        // Mostra a mensagem do usuário
        appendMessage(message, true);
        userInput.value = '';
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Mostra a resposta do chatbot
                appendMessage(data.response, false);
            } else {
                appendMessage("Desculpe, ocorreu um erro.", false);
            }
        } catch (error) {
            console.error('Erro:', error);
            appendMessage("Desculpe, ocorreu um erro.", false);
        }
    }
}

async function clearChat() {
    try {
        const response = await fetch('/clear-chat', {
            method: 'POST'
        });
        
        if (response.ok) {
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML = '';
        }
    } catch (error) {
        console.error('Erro ao limpar chat:', error);
    }
}

// Event Listeners
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

document.getElementById('close-chatbot').addEventListener('click', toggleChatbot);
document.getElementById('open-chatbot').addEventListener('click', toggleChatbot);

function moveSlide(direction) {
        const container = document.querySelector('.carousel-container');
        container.scrollBy({
            left: direction * 260,
            behavior: 'smooth'
        });
    }
