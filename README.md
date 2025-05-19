# AI-Powered Sign Language Translator

A real-time sign language translation web application that bridges communication between signers and non-signers using AI technology.

## Features

- 🎥 Live Sign Language to Text Translation
- 📝 Text-to-Sign Language Guide
- 🌐 Multi-language Translation Support
- 🔊 Text-to-Speech Capability

## Tech Stack

- **Frontend**: React
- **Backend**: FastAPI
- **Database**: MongoDB
- **AI/ML**: 
  - MediaPipe Hands for hand tracking
  - TensorFlow/Keras for sign recognition
- **APIs**:
  - Google Translate API
  - Google Text-to-Speech (gTTS)
  - WebRTC for camera access

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GOOGLE_TRANSLATE_API_KEY=your_api_key
MONGODB_URI=your_mongodb_uri
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   ├── routes/
│   │   └── services/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   └── package.json
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
