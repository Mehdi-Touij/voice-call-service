# N8N Voice Bot Service

Real-time voice AI assistant that integrates with N8N workflows, providing <200ms latency voice interactions.

## Features

- 🎙️ Real-time voice input via web browser
- 🔄 Speech-to-text using Deepgram
- 🤖 AI processing through N8N workflows (Claude Haiku)
- 🔊 Text-to-speech using ElevenLabs
- ⚡ Target latency: <200ms end-to-end
- 🌐 WebRTC powered by Daily.co

## Tech Stack

- **Backend**: FastAPI + Pipecat
- **STT**: Deepgram Nova 2
- **TTS**: ElevenLabs (Rachel voice)
- **WebRTC**: Daily.co
- **AI**: N8N workflow with Claude Haiku
- **Deployment**: Digital Ocean App Platform

## Quick Start

### Prerequisites

- GitHub account
- Digital Ocean account
- Active API keys for:
  - Deepgram
  - ElevenLabs  
  - Daily.co
  - N8N webhook URL

### Environment Variables

```bash
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/voice-session
DAILY_API_KEY=your_daily_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=your_voice_id
LOG_LEVEL=INFO
```

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/Mehdi-Touij/voice-call-service.git
cd voice-call-service
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your API keys

4. Run the server:
```bash
uvicorn main:app --reload
```

5. Open http://localhost:8000 in your browser

## Digital Ocean Deployment

### Quick Deploy

1. Fork this repository
2. Sign in to [Digital Ocean](https://www.digitalocean.com/)
3. Create new App → Connect GitHub → Select this repo
4. Set environment variables in DO dashboard
5. Deploy!

### Manual Configuration

The app is pre-configured with `.do/app.yaml` for optimal performance on Digital Ocean App Platform.

## Architecture

```
Browser → Voice Bot (DO) → Deepgram → N8N → Claude → ElevenLabs → Browser
```

### Flow:
1. User speaks into microphone
2. Audio streams to Daily.co WebRTC
3. Deepgram converts speech to text
4. Text sent to N8N webhook
5. N8N processes with Claude Haiku
6. Response converted to speech by ElevenLabs
7. Audio streams back to user

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /start_session` - Start voice session
- `POST /end_session/{session_id}` - End voice session
- `GET /session_status/{session_id}` - Check session status

## Performance Optimization

- Deepgram Nova 2 for fastest STT
- ElevenLabs streaming with lowest latency setting
- Persistent WebSocket connections
- Session cleanup after 5 minutes idle

## Troubleshooting

### Common Issues

1. **WebSocket connection fails**
   - Check Daily.co API key
   - Ensure browser allows microphone access

2. **High latency**
   - Verify all services are in same region
   - Check N8N webhook response time

3. **No audio output**
   - Verify ElevenLabs API key and voice ID
   - Check browser audio permissions

## License

MIT

## Support

For issues or questions, please open a GitHub issue.
