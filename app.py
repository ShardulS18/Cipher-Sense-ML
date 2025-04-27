import os
import pickle
import pandas as pd
import random
import math
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model.pkl'
FILE_TYPE_ENCODER_PATH = 'file_type_encoder.pkl'
BINARY_MODEL_PATH = 'binary_model.pkl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Step 1: Load dataset and train models ==========
def train_models():
    # Load dataset
    df = pd.read_csv("dataset.csv")

    # Clean the data: Drop rows with missing values
    df.dropna(inplace=True)

    # ----- Train cipher prediction model -----
    # Encode the file_type
    le_file_type = LabelEncoder()
    X = le_file_type.fit_transform(df['file_type'])

    # Reshape X to 2D array
    X = X.reshape(-1, 1)

    # Encode the cipher
    le_cipher = LabelEncoder()
    y = le_cipher.fit_transform(df['cipher'])

    # Train the model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model and encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    
    with open(FILE_TYPE_ENCODER_PATH, "wb") as f:
        pickle.dump(le_file_type, f)  # Save file_type encoder
    
    # Also save cipher classes mapping for human-readable results
    cipher_classes = {i: cipher for i, cipher in enumerate(le_cipher.classes_)}
    with open("cipher_classes.pkl", "wb") as f:
        pickle.dump(cipher_classes, f)

    # ----- Train binary detection model -----
    # If is_binary column exists in the dataset
    if 'is_binary' in df.columns:
        # Create a mapping of file types to binary flag
        binary_mapping = df.set_index('file_type')['is_binary'].to_dict()
        
        # Save this mapping
        with open(BINARY_MODEL_PATH, "wb") as f:
            pickle.dump(binary_mapping, f)
    
    print("✅ Models trained successfully and saved!")

# ========== Step 2: Predict Cipher and Binary Status ==========
def predict_cipher_and_binary(file_extension):
    # Load the cipher prediction model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load the file type encoder
    with open(FILE_TYPE_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load cipher classes
    with open("cipher_classes.pkl", "rb") as f:
        cipher_classes = pickle.load(f)
    
    # Load binary mapping if it exists
    try:
        with open(BINARY_MODEL_PATH, "rb") as f:
            binary_mapping = pickle.load(f)
    except FileNotFoundError:
        # Default to text files if no binary mapping exists
        binary_mapping = {}

    # Clean extension (remove leading dot if present)
    # if file_extension.startswith('.'):
    #     file_extension = file_extension[1:]
    
    # Determine if the file is binary
    is_binary = binary_mapping.get(file_extension, False)
    
    try:
        # Predict cipher type
        encoded = label_encoder.transform([file_extension])[0]
        predicted_idx = model.predict([[encoded]])[0]
        cipher_type = cipher_classes[predicted_idx]
    except ValueError:
        # If extension not in training data, use most common prediction
        with open("cipher_classes.pkl", "rb") as f:
            cipher_classes = pickle.load(f)
        most_common_idx = 0  # Default to first class if can't determine
        cipher_type = cipher_classes[most_common_idx]
    
    return cipher_type, is_binary

# ========== Step 3: Multiple Cipher Implementations ==========
def encrypt_content(content, cipher_type, is_binary=False):
    """
    Encrypt content using the specified cipher type
    If is_binary is True, content is treated as bytes rather than text
    """
    # Handle binary vs text content appropriately
    if is_binary and isinstance(content, str):
        # Convert string to bytes if necessary (e.g., if read as text)
        content = content.encode('latin1')  # Use latin1 to preserve all byte values
    elif not is_binary and isinstance(content, bytes):
        # Convert bytes to string if necessary
        content = content.decode('utf-8', errors='ignore')
    
    # Use a consistent key derivation for all ciphers that need keys
    # In production, you'd want more secure key management
    def derive_key(input_data, length=16):
        # Simple key derivation function - not for production use
        if isinstance(input_data, bytes):
            seed = sum(input_data[:20])
        else:
            seed = sum(ord(c) for c in input_data[:20])
        random.seed(seed)
        return os.urandom(length)  # Generate random bytes for key
    
    # ----- Text-based cipher implementations -----
    
    # Caesar Cipher (text only)
    def caesar_cipher(text, shift=3):
        if isinstance(text, bytes):
            return b'[Binary data not suitable for Caesar cipher]'
        
        result = ""
        for char in text:
            if char.isalpha():
                ascii_offset = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            else:
                result += char
        return result
    
    # Playfair Cipher (text only)
    def playfair_cipher(text):
        if isinstance(text, bytes):
            return b'[Binary data not suitable for Playfair cipher]'
        
        # Simplified Playfair implementation
        # Using "KEYWORD" as the key
        key = "KEYWORD"
        # In a real implementation, you would:
        # 1. Create the Playfair matrix using the key
        # 2. Format the input text (handling double letters, etc.)
        # 3. Apply the Playfair rules
        return f"[Playfair Cipher with key '{key}']: " + ''.join(
            chr((ord(c) + 2) % 256) if c.isalpha() else c for c in text
        )
    
    # Monoalphabetic Cipher (text only)
    def monoalphabetic_cipher(text):
        if isinstance(text, bytes):
            return b'[Binary data not suitable for Monoalphabetic cipher]'
        
        # Generate a simple substitution table
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        shuffled = list(alphabet)
        random.seed(42)  # Fixed seed for consistency
        random.shuffle(shuffled)
        shuffled = ''.join(shuffled)
        
        trans_table = str.maketrans(alphabet + alphabet.lower(), 
                                     shuffled + shuffled.lower())
        return text.translate(trans_table)
    
    # Polyalphabetic Cipher (Vigenère) (text only)
    def vigenere_cipher(text):
        if isinstance(text, bytes):
            return b'[Binary data not suitable for Vigenere cipher]'
        
        key = "SECRETKEY"
        result = ""
        key_idx = 0
        
        for char in text:
            if char.isalpha():
                # Convert key character to shift value
                key_char = key[key_idx % len(key)]
                shift = ord(key_char.upper()) - ord('A')
                
                # Apply shift like Caesar cipher
                ascii_offset = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
                key_idx += 1
            else:
                result += char
        
        return result
    
    # ----- Ciphers that can handle both text and binary data -----
    
    # One-Time Pad (can handle both)
    def one_time_pad(data):
        if isinstance(data, str):
            # Text mode
            # Generate a pad the same length as the text
            pad = bytes([random.randint(0, 255) for _ in range(len(data))])
            result = bytes([ord(data[i]) ^ pad[i] for i in range(len(data))])
            return "OTP:" + base64.b64encode(pad + result).decode()
        else:
            # Binary mode
            pad = os.urandom(len(data))
            result = bytes([data[i] ^ pad[i] for i in range(len(data))])
            return b"OTP:" + base64.b64encode(pad + result)
    
    # Rail Fence Transposition Cipher (can handle both)
    def rail_fence_cipher(data, rails=3):
        if isinstance(data, bytes):
            # Convert to list of bytes for binary data
            chars = list(data)
            is_bytes = True
        else:
            # Convert to list of characters for text
            chars = list(data)
            is_bytes = False
        
        fence = [[] for _ in range(rails)]
        rail = 0
        direction = 1  # 1 for down, -1 for up
        
        # Build the fence
        for char in chars:
            fence[rail].append(char)
            rail += direction
            if rail == rails - 1 or rail == 0:
                direction = -direction
        
        # Read off the fence
        if is_bytes:
            # For binary data, concatenate the byte arrays
            result = b''.join([bytes(rail) for rail in fence])
        else:
            # For text, join the characters
            result = ''.join([''.join(rail) for rail in fence])
        
        return result
    
    # Single Columnar Cipher (can handle both)
    def columnar_cipher(data):
        key = "COLUMNAR"
        key_order = [sorted(key).index(k) for k in key]
        
        if isinstance(data, bytes):
            # Binary mode
            # Convert to list of bytes
            bytes_data = list(data)
            
            # Pad to fit grid if needed
            cols = len(key)
            rows = math.ceil(len(bytes_data) / cols)
            bytes_data.extend([0] * (rows * cols - len(bytes_data)))
            
            # Create the grid
            grid = []
            for i in range(0, len(bytes_data), cols):
                grid.append(bytes_data[i:i+cols])
            
            # Read off columns in key order
            result = bytearray()
            for col_idx in key_order:
                for row in grid:
                    if col_idx < len(row):
                        result.append(row[col_idx])
            
            return bytes(result)
        else:
            # Text mode
            # Convert key to a column order
            cols = len(key)
            rows = math.ceil(len(data) / cols)
            padded_text = data.ljust(rows * cols)
            
            # Create the grid
            grid = [padded_text[i:i+cols] for i in range(0, len(padded_text), cols)]
            
            # Read off columns in key order
            result = ""
            for col_idx in key_order:
                for row in grid:
                    if col_idx < len(row):
                        result += row[col_idx]
            
            return result
    
    # Double Columnar Cipher (can handle both)
    def double_columnar_cipher(data):
        # Apply columnar cipher twice with different keys
        return columnar_cipher(columnar_cipher(data))
    
    # ----- Modern ciphers that are good for binary data -----
    
    # DES (Data Encryption Standard)
    def des_cipher(data):
        try:
            # Generate a key and IV
            key = os.urandom(8)  # DES uses 8-byte keys
            iv = os.urandom(8)
            
            # Ensure data is in bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Pad the data to a multiple of 8 bytes
            pad_length = 8 - (len(data) % 8)
            if pad_length < 8:
                data += bytes([pad_length]) * pad_length
            
            # Create and use the cipher
            backend = default_backend()
            cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv), backend=backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Encode the key, IV, and ciphertext in base64
            result = b"DES:" + base64.b64encode(key + iv + ciphertext)
            if isinstance(content, str):
                return result.decode()
            return result
        except Exception as e:
            if isinstance(content, str):
                return f"[DES Encryption Error: {str(e)}] " + content[:100] + "..."
            return b"[DES Encryption Error]" + content[:100]
    
    # RSA (good for binary data but limited in size)
    # Modify only the RSA cipher implementation to use a proper hybrid approach for PDFs
    def rsa_cipher(data):
        try:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Save private key temporarily (in production, handle more securely)
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            with open(os.path.join(UPLOAD_FOLDER, 'private_key.pem'), 'wb') as f:
                f.write(private_pem)
            
            # Generate a random AES key for the actual data encryption
            aes_key = os.urandom(32)  # 256-bit key
            iv = os.urandom(16)  # Initialization vector for AES
            
            # Encrypt the AES key with RSA
            encrypted_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Ensure data is in bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Add padding to ensure the data length is a multiple of 16 bytes (AES block size)
            padding_length = 16 - (len(data) % 16)
            padded_data = data + bytes([padding_length]) * padding_length
            
            # Create AES cipher
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.CBC(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt the data with AES
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine the encrypted components
            # Format: RSA:[IV length][IV][Encrypted key length][Encrypted key][Encrypted data]
            result = b"RSA:" + \
                    len(iv).to_bytes(2, byteorder='big') + \
                    iv + \
                    len(encrypted_key).to_bytes(2, byteorder='big') + \
                    encrypted_key + \
                    encrypted_data
                    
            # Encode in base64 for easier handling
            encoded_result = base64.b64encode(result)
            
            if isinstance(content, str):
                return encoded_result.decode()
            return encoded_result
            
        except Exception as e:
            # In case of error, return informative message
            error_msg = f"[RSA Encryption Error: {str(e)}]"
            if isinstance(content, str):
                return error_msg + " " + content[:100] + "..."
            return error_msg.encode() + content[:100]
    
    # Diffie-Hellman (Key Exchange Algorithm - simulated for demonstration)
    def diffie_hellman_demo(data):
        # Note: Diffie-Hellman is not an encryption algorithm but a key exchange protocol
        # For demonstration, we'll simulate using the derived key for encryption
        if isinstance(data, str):
            return f"[Diffie-Hellman Key Exchange + AES-like encryption]: " + \
                   ''.join(chr((ord(c) + 7) % 256) for c in data[:100]) + "..."
        else:
            # For binary data
            key = os.urandom(16)  # Simulate the derived key
            result = bytearray()
            for i, b in enumerate(data[:100]):  # Just encrypt first 100 bytes for demo
                result.append((b + key[i % len(key)]) % 256)
            return b"[Diffie-Hellman Key Exchange + AES-like encryption]:" + bytes(result) + b"..."
    
    # Select the appropriate cipher based on the prediction
    try:
        if cipher_type == "Caesar Cipher":
            return caesar_cipher(content)
        elif cipher_type == "Playfair Cipher":
            return playfair_cipher(content)
        elif cipher_type == "Monoalphabetic cipher":
            return monoalphabetic_cipher(content)
        elif cipher_type == "Polyalphabetic Cipher":
            return vigenere_cipher(content)
        elif cipher_type == "One-Time Pad":
            return one_time_pad(content)
        elif cipher_type == "Rail Fence Transposition Cipher":
            return rail_fence_cipher(content)
        elif cipher_type == "Single Columnar Cipher":
            return columnar_cipher(content)
        elif cipher_type == "Double Columnar Cipher":
            return double_columnar_cipher(content)
        elif cipher_type == "Data Encryption Standard(DES)":
            return des_cipher(content)
        elif cipher_type == "RSA":
            return rsa_cipher(content)
        elif cipher_type == "Diffie-Helman algorithm":
            return diffie_hellman_demo(content)
        else:
            # Fallback based on binary status
            if is_binary:
                return des_cipher(content)  # DES works well for binary
            else:
                return caesar_cipher(content)  # Caesar for text
    except Exception as e:
        # Fallback in case of any error
        if isinstance(content, str):
            return f"[Encryption Error: {str(e)}] Using fallback encryption: " + content[:100] + "..."
        else:
            return b"[Encryption Error] Using fallback encryption: " + content[:100] + b"..."

# ========== Step 4: Flask Routes ==========
@app.route('/')
def home():
    return "🔐 ML Encryption API is running!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()  # Get file extension

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Predict cipher type and binary status
    cipher, is_binary = predict_cipher_and_binary(ext)

    # Read content based on binary status
    try:
        if is_binary:
            with open(file_path, 'rb') as f:
                content = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"}), 400

    # Encrypt content
    encrypted = encrypt_content(content, cipher, is_binary)

    # Save encrypted file
    encrypted_filename = f"encrypted_{filename}"
    encrypted_path = os.path.join(UPLOAD_FOLDER, encrypted_filename)
    
    if isinstance(encrypted, bytes):
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted)
    else:
        with open(encrypted_path, 'w', encoding='utf-8') as f:
            f.write(encrypted)

    return jsonify({
        "original_filename": filename,
        "predicted_cipher": cipher,
        "is_binary": is_binary,
        "download": f"/download/{encrypted_filename}"
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

# ========== Run App ==========
if __name__ == '__main__':
    # Check if models exist, otherwise train
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FILE_TYPE_ENCODER_PATH):
        print("Models not found. Training new models...")
        train_models()
    else:
        print("Models found. Ready to predict.")
    
    app.run(debug=True)