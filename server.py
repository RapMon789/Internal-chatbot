from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from IPython.display import Markdown
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.tools.tavily_search import TavilySearchResults
import os, json
from datetime import datetime
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
from langdetect import detect

os.environ['TAVILY_API_KEY'] = 'YOUR_TAVILY_API_KEY'

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SECRET_KEY'] = 'YOUR_SECRET_KEY'

if not os.listdir('instance'):
    with app.app_context():
        db.create_all()
        
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

memory = ConversationBufferMemory(return_messages=True, memory_key='chat_history')
runnable = RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter('chat_history'))
chat_history = memory.load_memory_variables({})['chat_history']

class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')
        

class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


def create_vectorstore():

    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    loader = DirectoryLoader("data")
    documents = loader.load()
    
    print(f"{len(documents)} PDF pages loaded.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    texts=text_splitter.split_documents(documents)
    
    print(f"{len(texts)} chunks created.")
    vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(model_name=model_name),persist_directory="vectorstores/db/")      
    
    return vectorstore


def update_vectorstore():
    dossier_data = "data"

    pdf_files = []

    for f in os.listdir(dossier_data):
        if f.endswith(".pdf"):
            pdf_files.append(f)

    with open("check.txt", "r") as txt_file:
        files_in_txt = txt_file.read().splitlines()

    missing_files = []
    deleted_files = []

    for f in pdf_files:
        if f not in files_in_txt:
            missing_files.append(f)

    for f in files_in_txt:
        if f not in pdf_files:
            deleted_files.append(f)

    if missing_files or deleted_files:

        if deleted_files:
            with open("check.txt", "w") as txt_file:
                for f in pdf_files:
                    txt_file.write(f + "\n")
            print("File(s) deleted: ")
            for f in deleted_files:
                print(f)
            print("Updating vector store.")
            vectorstore = Chroma(embedding_function=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf") ,persist_directory="vectorstores/db/")

            for title in deleted_files:
                test = vectorstore.get(where={'source':f'data\\{title}'})['ids']
                vectorstore.delete(ids=test)

        if missing_files: 
            with open("check.txt", "w") as txt_file:
                for f in pdf_files:
                    txt_file.write(f + "\n")
            print("New file(s) detected: ")
            for f in missing_files:
                print(f)
            print("Updating vector store.")
            vectorstore = create_vectorstore() 
            print("")

        return vectorstore

    else:
        vectorstore = Chroma(embedding_function=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf") ,persist_directory="vectorstores/db/")
        return vectorstore
        

def web_choice(model_choice, query, chat_history):
    model_json = Ollama(model=model_choice, format='json', base_url="http://127.0.0.1:11434", verbose=True, temperature=0)

    if os.listdir('data'):
        relevant_docs = vectorstore.similarity_search_with_relevance_scores(query, k=10)
        print(relevant_docs[0][0])
        threshold = 0.1
        if relevant_docs[0][1] > threshold:

            check_base = """
            You are an expert at checking if a provided content can answer a user question.
            Give a binary choice 'yes' or 'no' based on the question. 
            Return the JSON with a single key 'choice' with no preamble or explanation. 

            Question: {question} 

            Content: {content}
            """

            check_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", check_base),
                ("human", "{question}")
            ]
            )
            is_really_relevant = check_prompt | model_json | JsonOutputParser()

            contents = [doc[0].page_content for doc in relevant_docs]

            check_choice = is_really_relevant.invoke({"question": query, "content": contents})['choice']
            print("check_choice: ", check_choice)

            if check_choice == 'yes':
                return {'choice': 'generate', 'relevant_docs': relevant_docs}

    router_base = f"""
        You are an expert at routing a user question to either the generation stage or web search. 
        Before doing anything, YOU MUST check the chat history to get the context of the question if there is one.
        If there is any reference in the question about special date, you must answer 'web_search'.
        If there is any reference about someone, you must answer 'web_search'.
        Use the chat history to understand the context and determine if the question is about a recent event or requires more context for a better answer.
        The answer may already be in the chat history. If yes, give the binary choice 'generate' and give the answer from the chat history.
        Use the web search for questions that require more context for a better answer, or recent events.
        Otherwise, you can skip and go straight to the generation phase to respond.
        You do not need to be stringent with the keywords in the question related to these topics.
        Give a binary choice 'web_search' or 'generate' based on the question.
        Return the JSON with a single key 'choice' with no preamble or explanation. 
        
        Question to route: {query} 

        Chat history: {chat_history}
        """
    
    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", router_base),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    question_router = runnable | router_prompt | model_json | JsonOutputParser()
    return question_router.invoke({"question": query, "chat_history": chat_history})


def web_search(model_choice, query, chat_history):
    model_json = Ollama(model=model_choice, format='json', base_url="http://127.0.0.1:11434", verbose=True, temperature=0)

    query_base = f"""
        You are an expert at crafting web search queries for research questions.
        More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format. 
        Reword their query to be the most effective web search string possible. 
        DO NOT CHANGE WORDS. If the query mentions someone, somewhere or something, keep it as it is, it must be exactly the same.
        Return the JSON with a single key 'query' with no premable or explanation. 

        Question to transform: {query} 
     """

    query_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_base),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

    query_chain = query_prompt | model_json | JsonOutputParser()

    # Web Search Tool
    #wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
    #web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
    
    return TavilySearchResults(k=5), query_chain.invoke({"question": query, "chat_history": chat_history})["query"]


def generate_answer_chain(model_choice, language, context):
    
    model = Ollama(model=model_choice, base_url="http://127.0.0.1:11434", verbose=True, temperature=0) 

    generate_base = f"""
       You are a helpful chatbot. Just answer the query. DO NOT mention previous conversation when it is not required.

       You MUST answer ONLY in {language}.
       Web Search Context: {context} 
     """

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", generate_base),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )

    # Chain
    generate_chain = runnable | generate_prompt | model | StrOutputParser()
    
    return generate_chain


def run_agent(language, query, model):
    
    print("Query: ", query)
    print("Language: ", language)  
    print("model_choice: ", model)

    search_result = ""



    output = web_choice(model, query, chat_history)
    print(output['choice']) 
        
    if output['choice'] == 'generate':
        if 'relevant_docs' in output:
            relevant_docs = output['relevant_docs']
            retrieved_contents = [doc[0].page_content for doc in relevant_docs]
        else:
            retrieved_contents = []
            
        generate_chain = generate_answer_chain(model, language, retrieved_contents)
        generation = generate_chain.invoke({"question": query, "chat_history": chat_history})
    
    elif output['choice'] == "web_search":
        web_search_tool, search_query = web_search(model, query, chat_history)
        print("Step: Optimizing Query For Web Search: ", search_query)
        search_result = web_search_tool.invoke(search_query)

        contents = []
        for item in search_result:
            contents.append((item['content']))

        print("Step: Generating Final Response")
        generate_chain = generate_answer_chain(model, language, contents)
        generation = generate_chain.invoke({"question": query, "chat_history": chat_history})

    markdown_obj = Markdown(generation)     
    
    answer = markdown_obj.data
    date = datetime.now().isoformat()

    with open('logs.json') as f:
        logs = json.load(f)

    new = {
            "input": query,
            "output": answer,
            "date": date,
            "context": search_result
        }
    
    logs.append(new)

    with open("logs.json", "w") as f:
        json.dump(logs, f, indent=2)

    return(markdown_obj.data, date)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm() 

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/loadMemory', methods=['POST'])
def loadMemory():
     
    print("1")
    memory.clear()
    print("2")
    with open('logs.json') as f:
        logs = json.load(f)
    print("3")
    for log in logs:
        print('log: ', log['input'])
        memory.save_context({'input': log['input']}, {'output': log['output']})
    print("4")
    return jsonify({'response': "True"})


@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    memory.clear()

    files = os.listdir("audio")
    for file in files:
        file_path = os.path.join("audio", file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
            
    return jsonify({'success': True})

@app.route('/chat', methods=['POST'])
def chat():
    language = request.json.get('lang')
    user_input = request.json.get('message')
    model_choice = request.json.get('model')

    if user_input:
        response, timestamp = run_agent(language, user_input, model_choice)    
        memory.save_context({'input' : user_input}, {'output': response})
        
        audio_filename = f"response_audio_{timestamp}.mp3".replace(":", "")
        audio_filename = audio_filename.replace(".", "", 1)
        audio_filename = audio_filename.replace("-", "")
        
        audio_response = gTTS(text=response, lang=detect(response))
        audio_response.save(f"audio/{audio_filename}")
        
        return jsonify({'response': response, 'audio_filename': audio_filename})


@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(directory='audio', path=filename)

@app.route('/test') 
def hello_world():
    return "<p>Hello, World! This is a test.</p>"

vectorstore = update_vectorstore()
print("vectorstore successfully update.")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=80, debug=True)