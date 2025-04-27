# CipherSense: Adaptive Encryption 🔐

## Overview

CipherSense is an intelligent encryption system that uses machine learning to analyze file types and recommend the optimal encryption method for each file. By bridging the gap between advanced encryption techniques and everyday usability, CipherSense ensures that every file receives the most appropriate level of protection without requiring technical expertise from the user.

## Problem Statement ❓

Most encryption solutions apply a one-size-fits-all approach to file security, leading to:
- Inefficient encryption for certain file types
- Potentially weak protection for sensitive files
- Technical complexity that discourages adoption

CipherSense addresses these challenges by automatically determining the best encryption method based on the specific characteristics of each file.

## Key Features ✨

- **Intelligent Encryption Selection**: Uses machine learning to recommend the most suitable encryption method
- **File Type Analysis**: Automatically identifies whether files are binary or text-based
- **Versatile Encryption Options**: Supports both classical ciphers (Caesar, Vigenère) and modern methods (DES, RSA)
- **User-Friendly Interface**: Simple, intuitive design requiring no technical expertise
- **Efficient Processing**: Optimized for both speed and security

## How It Works 🛠️

1. **File Upload**: Users upload a file through the intuitive interface
2. **Analysis**: CipherSense identifies the file extension and determines if it's binary or text
3. **Prediction**: A trained Random Forest Classifier predicts the optimal cipher for the file type
4. **Encryption**: The system applies the recommended encryption method
5. **Secure Output**: The encrypted file is provided to the user

## Technical Architecture 🏗️

### Frontend
- Built with React.js
- Provides intuitive user interaction
- Handles file uploads and displays recommendations

### Backend
- Developed using Flask
- Processes file analysis
- Implements various encryption algorithms
- Hosts the machine learning model

### Machine Learning Component 🧠
- Uses a Random Forest Classifier
- Formula: ŷ = mode(h₁(x), h₂(x), ..., hₙ(x))
- Where h_i(x) is the prediction of the ith decision tree and ŷ is the final predicted cipher type
- Multiple decision trees vote on the best encryption method for each file type

## Installation 💻

```bash
# Clone the repository
git clone https://github.com/yourusername/ciphersense.git

# Navigate to the project directory
cd ciphersense

# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
pip install -r requirements.txt

# Start the application
python app.py
```

## Usage 📋

1. Open your browser and navigate to `http://localhost:5000`
2. Use the file upload interface to select a file for encryption
3. CipherSense will analyze the file and recommend an encryption method
4. Review and confirm the recommendation
5. Download the encrypted file

## Future Development 🚀

- Additional encryption algorithms
- Support for more file types
- Batch processing capabilities
- Encrypted file management system
- Integration with cloud storage providers
