# NDA Analysis Assistant

A Streamlit-based web application that uses AI to analyze Non-Disclosure Agreements (NDAs) and provide comprehensive insights through chat interaction.

## Features

- 📄 **PDF Document Upload**: Upload NDA documents in PDF format
- 🤖 **AI-Powered Analysis**: Comprehensive NDA analysis using GPT models
- 💬 **Interactive Chat**: Ask questions about your NDA and get instant answers
- 🔍 **Smart Intent Classification**: Automatically determines whether to summarize or answer questions
- 📋 **Structured Analysis**: Covers all key NDA components (parties, obligations, terms, etc.)
- 💭 **Conversation Memory**: Maintains chat history throughout your session
- 🎯 **Source Attribution**: Shows which document sections were used for answers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nda-chatbot.git
cd nda-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Enter API Key**: Add your OpenAI API key in the sidebar
2. **Upload NDA**: Upload a PDF NDA document
3. **Start Chatting**: Ask questions or request comprehensive analysis

### Example Queries:
- "Analyze this NDA comprehensively"
- "Who are the parties involved?"
- "What are the confidentiality obligations?"
- "How long does this NDA last?"
- "Are there any unusual clauses?"

## Configuration

The app supports various configuration options:

- **Model Selection**: Default is GPT-4, can be modified in `nda_chatbot.py`
- **Memory Settings**: Conversation memory keeps last 10 exchanges
- **Chunk Strategy**: Automatically selects between "stuff" and "map_reduce" based on document size

## Project Structure

```
nda-chatbot/
├── app.py                 # Main Streamlit application
├── nda_chatbot.py        # Core NDA analysis chatbot class
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore file
└── .streamlit/          # Streamlit configuration (optional)
    └── config.toml
```

## Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your OpenAI API key as a secret in Streamlit Cloud settings

### Other Platforms
The app can be deployed on:
- Heroku
- AWS
- Google Cloud Platform
- Azure

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

