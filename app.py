import os
import numpy as np
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import re

# OpenAI imports
import openai
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your existing skin disease model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'Models/skin_disease_model.h5'))

# OpenAI Configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

# Store conversation history (in production, consider using a database)
conversation_sessions = {}

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])

def clean_markdown_formatting(text):
    """Remove markdown formatting from text"""
    # Remove bold formatting (**text** and *text*)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove italic formatting (__text__ and _text_)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove code formatting (`text`)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove headers (# ## ###)
    text = re.sub(r'^#+\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
    
    return text

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def predict(filename, model):
    classes = ['Acne On Body', 'Acne On Face', 'Acne On Forehead', 'Actinic Cheilitis',
               'Alopecia Areata', 'Eczema Foot', 'Nail Fungal', 'Nose Rosacea', 'Raynauds Phenomenon']
    img = cv2.resize(cv2.imread(filename), (32, 32)) / 255.0
    prediction = model.predict(img.reshape(1, 32, 32, 3))
    return classes[np.argmax(prediction)]

# Your existing routes
@app.route('/')
@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/check')
def check():
    return render_template("check.html")

@app.route('/disease')
def disease():
    return render_template("disease.html")

@app.route('/aa')
def aa():
    return render_template("diseases/aa.html")

@app.route('/ac')
def ac():
    return render_template("diseases/ac.html")

@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/aob')
def aob():
    return render_template("diseases/aob.html")

@app.route('/aof')
def aof():
    return render_template("diseases/aof.html")

@app.route('/aofh')
def aofh():
    return render_template("diseases/aofh.html")

@app.route('/ef')
def ef():
    return render_template("diseases/ef.html")

@app.route('/nf')
def nf():
    return render_template("diseases/nf.html")

@app.route('/nr')
def nr():
    return render_template("diseases/nr.html")

@app.route('/rp')
def rp():
    return render_template("diseases/rp.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/uploads')
    if request.method == 'POST':
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result = predict(img_path, model)
            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, class_result=class_result)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')

# NEW: OpenAI Chat API Routes
@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handle chat messages and return OpenAI responses"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Check if OpenAI API key is configured
        if not openai.api_key:
            return jsonify({
                'error': 'Chat service is not configured. Please contact support.'
            }), 503
        
        # Initialize conversation history for new sessions
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
        
        # Add user message to conversation history
        conversation_sessions[session_id].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Prepare messages for OpenAI API
        messages = [
            {
                'role': 'system',
                'content': """You are Skeeze, a specialized AI assistant ONLY for skin health and dermatology questions. 
                You work alongside an AI-powered skin disease detection system that can analyze images and identify conditions like:
                - Acne (on body, face, forehead)
                - Actinic Cheilitis
                - Alopecia Areata
                - Eczema Foot
                - Nail Fungal infections
                - Nose Rosacea
                - Raynaud's Phenomenon
                
                STRICT GUIDELINES:
                - ONLY answer questions related to skin health, dermatology, skincare, skin conditions, and skin treatments
                - If someone asks about anything NOT related to skin/dermatology (like locations, general knowledge, other topics), politely redirect them by saying: "I'm specifically designed to help with skin health questions. Please ask me about skin conditions, skincare routines, or dermatology topics. For other questions, please use a general search engine or assistant."
                - Always be supportive, empathetic, and understanding for skin-related queries
                - Provide accurate, evidence-based information about skin health only
                - For serious symptoms or concerns, always recommend consulting healthcare professionals
                - Do not provide specific medical diagnoses - that's what our image analysis tool is for
                - Focus on general advice, prevention, skincare routines, and when to seek medical help
                - You can mention that users can upload images using our "Check" feature for AI-powered analysis
                - Be concise but thorough in your responses
                - Do NOT use markdown formatting like *bold* or **bold** - use plain text only
                - Write in natural, conversational language without special formatting
                - If asked about the diseases you can detect, mention the 9 conditions listed above"""
            }
        ]
        
        # Add conversation history (limit to last 20 messages to manage token usage)
        recent_history = conversation_sessions[session_id][-20:]
        for msg in recent_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Call OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                messages=messages,
                max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', 500)),
                temperature=float(os.getenv('OPENAI_TEMPERATURE', 0.7)),
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            assistant_message = response.choices[0].message.content.strip()
            
            # Clean up markdown formatting
            assistant_message = clean_markdown_formatting(assistant_message)
            
            # Add assistant response to conversation history
            conversation_sessions[session_id].append({
                'role': 'assistant',
                'content': assistant_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Log the interaction
            logger.info(f"Chat interaction - Session: {session_id}, User: {user_message[:50]}...")
            
            return jsonify({
                'success': True,
                'message': assistant_message,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except openai.error.RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            return jsonify({
                'error': 'Service temporarily unavailable due to high demand. Please try again in a moment.'
            }), 429
            
        except openai.error.InvalidRequestError as e:
            logger.error(f"Invalid OpenAI request: {str(e)}")
            return jsonify({
                'error': 'Invalid request. Please try rephrasing your message.'
            }), 400
            
        except openai.error.AuthenticationError:
            logger.error("OpenAI authentication failed")
            return jsonify({
                'error': 'Service configuration error. Please contact support.'
            }), 500
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return jsonify({
                'error': 'Unable to process your request at the moment. Please try again later.'
            }), 500
            
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/api/chat/history/<session_id>')
def get_chat_history(session_id):
    """Get conversation history for a session"""
    try:
        history = conversation_sessions.get(session_id, [])
        return jsonify({
            'success': True,
            'history': history,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return jsonify({'error': 'Unable to fetch chat history'}), 500

@app.route('/api/chat/clear/<session_id>', methods=['POST'])
def clear_chat_history(session_id):
    """Clear conversation history for a session"""
    try:
        if session_id in conversation_sessions:
            del conversation_sessions[session_id]
        return jsonify({
            'success': True,
            'message': 'Chat history cleared',
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': 'Unable to clear chat history'}), 500

# Health check endpoint for monitoring
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'skin_disease_model': 'loaded' if model else 'error',
            'openai_configured': 'yes' if openai.api_key else 'no'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)