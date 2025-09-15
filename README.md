# Ploop AI Server

AI Server for the Ploop 2025 Solution Challenge

## Features

- **AI Chat Interface**: Process AI chat requests with customizable models and parameters
- **Ploop Analysis**: Specialized data analysis for Ploop-specific use cases
- **Health Monitoring**: Built-in health check endpoints
- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Docker Support**: Containerized deployment ready

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd Ploop_AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t ploop-ai-server .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here ploop-ai-server
```

## API Endpoints

### Health Check
- `GET /` or `GET /health` - Server health status

### AI Chat
- `POST /ai/chat` - Process AI chat requests
  ```json
  {
    "prompt": "Your question here",
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000,
    "temperature": 0.7
  }
  ```

### Ploop Analysis
- `POST /ploop/analyze` - Perform Ploop-specific analysis
  ```json
  {
    "data": {...},
    "analysis_type": "general"
  }
  ```

### Configuration
- `GET /config` - Get server configuration

## API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for AI features | - |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `DEBUG` | Debug mode | false |
| `LOG_LEVEL` | Logging level | INFO |

## Development

The server is built with FastAPI and includes:
- Automatic request/response validation with Pydantic
- CORS middleware for cross-origin requests
- Structured logging
- Error handling
- Type hints throughout

## License

This project is part of the Ploop 2025 Solution Challenge.
