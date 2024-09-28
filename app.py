from flask import Flask, request, render_template, redirect, url_for, session
import os
import uuid
from RAG_FUNCTIONS.RAG_creation import (
    load_and_split_pdf,
    create_and_persist_vectorstore,
    setup_contextual_qa_chain,
    setup_conversational_chain,
    llm,
)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure random key

# Global store for conversational RAG chains
rag_chains = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'pdf_file' not in request.files:
        return redirect(url_for('home'))

    uploaded_file = request.files['pdf_file']
    session_id = f"session_{os.path.splitext(uploaded_file.filename)[0]}_{uuid.uuid4()}"

    # Process the uploaded PDF
    splits = load_and_split_pdf(uploaded_file)
    retriever = create_and_persist_vectorstore(splits, uploaded_file.filename)
    
    # Setup RAG chain
    rag_chain = setup_contextual_qa_chain(llm, retriever)
    rag_chains[session_id] = rag_chain  # Store the RAG chain in a global dictionary

    session['conversation_history'] = []
    session['session_id'] = session_id
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session_id = session.get('session_id')
    conversational_rag_chain = rag_chains.get(session_id)

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        conversation_history = session.get('conversation_history', [])

        if conversational_rag_chain and user_input:
            # Create chat history in the required format (list of messages)
            messages = []
            for q, a in conversation_history:
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})

            # Append the current user input
            messages.append({"role": "user", "content": user_input})

            # Pass the messages as the chat_history
            response = conversational_rag_chain.invoke(
                {
                    "input": user_input,  # This should be just the user input
                    "chat_history": messages,  # Now we pass the formatted messages
                    "context": "Any additional context if needed"  # Add context if necessary
                },
                config={"configurable": {"session_id": session_id}}
            )
            answer = response['answer']
            conversation_history.append((user_input, answer))
            session['conversation_history'] = conversation_history

    return render_template('chat.html', conversation_history=session.get('conversation_history', []))

if __name__ == '__main__':
    app.run(debug=True)
