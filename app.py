import os
from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__, template_folder='templates')

# Initialize the chatbot matching engine
FAQS_PATH = os.path.join(os.path.dirname(__file__), 'faqs.json')
chatbot = FAQChatbot(FAQS_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'response': 'Please say something!'}), 400
            
        # Get matching answer
        response = chatbot.get_response(message)
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Running on local port 5002 for Task 2
    app.run(debug=True, port=5002)
