# internal network chatbot project
This project implements a versatile chatbot system designed to operate within an internal network environment. It combines natural language processing, web search capabilities, and local document retrieval to provide intelligent responses to user queries.

## Features:
Multi-model support: Utilizes Ollama for different language models<br />
Web search integration: Uses DuckDuckGo and Tavily for up-to-date information retrieval<br />
Local document search: Implements vector store using Chroma for efficient document retrieval<br />
Conversation memory: Maintains context using ConversationBufferMemory<br />
Load memory: Loads previous conversations thanks to the logs file<br />
User authentication: Includes login and registration functionality<br />
Multi-language support: Detects and responds in various languages<br />
Audio response: Generates audio responses using gTTS<br />
Web interface: Provides a user-friendly dashboard for interaction

## Technologies Used:
Flask: Web framework for the backend<br />
SQLAlchemy: ORM for database management<br />
LangChain: For building language model chains and tools<br />
Chroma: Vector store for document embedding and retrieval<br />
Ollama: Local language model integration<br />
gTTS: Text-to-speech conversion<br />

## Setup and Installation:
Clone the repository<br />
Install the required dependencies: pip install -r requirements.txt <br />
Make sure that the structure of the project looks like this:<br />

# chatbot-internal-network

* [audio/](./audio/)
* [data/](./data/)
* [static/](./static/)
  * [favicon.ico](./static/favicon.ico)
  * [styles.css](./static/styles.css)
  * [script.js](./static/script.js)
* [templates/](./templates/)
  * [index.html](./templates/index.html)
  * [home.html](./templates/home.html)
  * [login.html](./templates/login.html)
  * [register.html](./templates/register.html)
* [check.txt](./check.txt)
* [logs.json](./logs.json)
* [README.md](./README.md)
* [requirements.txt](./requirements.txt)
* [server.py](./server.py)

Set up Ollama with the desired language models (make sure that your Ollama is running by searching the IP address 127.0.0.1:11434 in your browser.)<br />
Configure the Tavily API key in the environment variables (If you do not want to use any API key, you can use istead the DuckDuckGo search.<br />
Do not forget to put the web_search_tool in the return instead of the actual `TavilySearchResults(k=5)`.)<br />
Run the Flask application: python server.py<br />
Search in your browser the IP address of your computer, it will be written in the terminal as "Running on http://YOUR_IP_ADDRESS:80".

## Licenses:
- BSD-3-Clause
- MIT 
- Apache-2.0<br />

Raphaël Monnier - Researcher (IT-AI/R&D) - Emotionwave
