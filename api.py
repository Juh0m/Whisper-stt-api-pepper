from flask import Flask, request, jsonify
import os
import whisper
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'aac'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def upload_audio():
    """
    Transcription endpoint for .aac audio files
    Expects a file in the 'audio' field of the form data
    """
    try:
        # Check if file is in request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only .aac files are allowed'}), 400
        
        filename = "audio.aac"
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Transcribe the file (with Whisper)
        model = whisper.load_model("turbo", device="cuda")
        result = model.transcribe("uploads/audio.aac", language="en")
        print(result["text"])

        # Return transcription in response
        return jsonify({
            'text': result["text"],
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'}), 413

if __name__ == '__main__':
    print(f"Starting STT Transcription API...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Max file size: {MAX_FILE_SIZE // (1024*1024)}MB")
    print(f"Server running on http://0.0.0.0:5000")
    
    app.run(debug=False, host='0.0.0.0', port=5000) #REPLACE WITH YOUR IP
    
    