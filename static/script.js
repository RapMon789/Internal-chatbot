async function sendMessage() {
    const langChoice = document.getElementById('lang').value;
    const userInput = document.getElementById('user-input').value;
    const modelChoice = document.getElementById('models').value;
    if (!userInput.trim()) return;

    const messageContainer = document.getElementById('message-container');

    const userMessageWrapper = document.createElement('div');
    userMessageWrapper.classList.add('message');

    const userMessage = document.createElement('div');
    userMessage.classList.add('user-message');
    userMessage.textContent = `You: ${userInput}`;
    userMessageWrapper.appendChild(userMessage);
    messageContainer.appendChild(userMessageWrapper);
    messageContainer.scrollTop = messageContainer.scrollHeight;
    messageContainer.scroll;
    document.getElementById('user-input').value = '';
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ lang: langChoice, message: userInput, model: modelChoice })
        });

        const data = await response.json();

        const botMessageWrapper = document.createElement('div');
        botMessageWrapper.classList.add('message');

        const botMessage = document.createElement('div');
        botMessage.classList.add('bot-message');
        botMessage.textContent = `Bot: ${data.response}`;

        const loudspeakerButton = document.createElement('button');
        loudspeakerButton.classList.add('loudspeaker-button');
        loudspeakerButton.innerHTML = '&#128264;'; 
        loudspeakerButton.onclick = () => {
            const audio = new Audio(`audio/${data.audio_filename}`);
            audio.play();
        };

        botMessageWrapper.appendChild(botMessage);
        botMessageWrapper.appendChild(loudspeakerButton);
        messageContainer.appendChild(botMessageWrapper);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        messageContainer.scroll
        
    } catch (error) {
        console.error('Error:', error);
        alert('There was an error communicating with the chatbot.');
    }
}

async function loadMemory() {
    const messageContainer = document.getElementById('message-container');
    try {
        const response = await fetch('/loadMemory', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        });
        
        await response.json();

        const botMessageWrapper = document.createElement('div');
        botMessageWrapper.classList.add('message');
        const botMessage = document.createElement('div');
        botMessage.classList.add('bot-message');
        botMessage.textContent = "Bot: Memory loaded successfully !";

        botMessageWrapper.appendChild(botMessage);
        messageContainer.appendChild(botMessageWrapper);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        messageContainer.scroll
        
    } catch (error) {
        console.error('Error:', error);
        alert('There was an error with the memory loading.');
    }
}

function clearMemory() {
    fetch('/clear_memory', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log('Memory cleared:', data.success);
    })
    .catch(error => {
        console.error('Error clearing memory:', error);
    });
}

window.onbeforeunload = clearMemory;