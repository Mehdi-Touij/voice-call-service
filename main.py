#!/usr/bin/env python3

import asyncio
import logging
import os
import sys
from typing import Async

# Pipecat imports
from pipecat.frames.frames import Frame, AudioRawFrame, TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

# Web framework for N8N integration
from aiohttp import web, web_request
import aiohttp_cors
import json
import httpx

# Configure logging
logging.basicConfig(format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "aD6riP1btT197c6dACmy")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
PORT = int(os.getenv("PORT", 8080))

class N8NProcessor:
    """Custom processor to integrate with your N8N AI Agent"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session_id = f"voice_{asyncio.get_event_loop().time()}"
        
    async def process_text(self, text: str) -> str:
        """Send text to N8N and get AI response"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {
                    "message": text,
                    "sessionId": self.session_id,
                    "channel": "voice_call",
                    "timestamp": str(asyncio.get_event_loop().time())
                }
                
                logger.info(f"Sending to N8N: {text}")
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                ai_response = result.get("output", result.get("response", result.get("message", "I'm sorry, I didn't understand that.")))
                
                logger.info(f"N8N response: {ai_response}")
                return ai_response
                
        except Exception as e:
            logger.error(f"N8N processing error: {e}")
            return "I'm experiencing some technical difficulties. Please try again."

class VoiceBot:
    """Main voice bot using Pipecat framework"""
    
    def __init__(self):
        self.n8n_processor = N8NProcessor(N8N_WEBHOOK_URL)
        self.current_task = None
        
    def create_pipeline(self, transport):
        """Create the voice processing pipeline"""
        
        # 1. Speech-to-Text (Deepgram)
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2",
            language="en-US",
            interim_results=True,
            smart_format=True,
            utterance_end_ms=1000,
            vad_events=True
        )
        
        # 2. Voice Activity Detection (Silero)
        vad = SileroVADAnalyzer()
        
        # 3. Text-to-Speech (ElevenLabs via Cartesia for better streaming)
        tts = CartesiaTTSService(
            api_key=ELEVENLABS_API_KEY,  # We'll use ElevenLabs
            voice_id=ELEVENLABS_VOICE_ID,
            model="sonic-english",  # Fast model
            sample_rate=16000
        )
        
        # Create the pipeline
        pipeline = Pipeline([
            transport.input(),   # Audio input from WebRTC
            vad,                 # Voice activity detection
            stt,                 # Speech to text
            self.llm_processor,  # Custom LLM processor (N8N integration)
            tts,                 # Text to speech
            transport.output(),  # Audio output to WebRTC
        ])
        
        return pipeline
    
    async def llm_processor(self, frame: Frame):
        """Process LLM requests through N8N"""
        if isinstance(frame, TranscriptionFrame):
            user_text = frame.text.strip()
            if user_text:
                logger.info(f"User said: {user_text}")
                
                # Get response from N8N AI Agent
                ai_response = await self.n8n_processor.process_text(user_text)
                
                # Return text frame for TTS
                return TextFrame(ai_response)
        
        return frame
    
    async def start_voice_session(self, room_url: str):
        """Start a voice session with given room URL"""
        try:
            # Configure transport (Daily.co for WebRTC)
            transport = DailyTransport(
                room_url,
                None,  # No token needed for temporary rooms
                "Voice Assistant",
                DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    transcription_enabled=False,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),
                )
            )
            
            # Create and run pipeline
            pipeline = self.create_pipeline(transport)
            task = PipelineTask(pipeline, PipelineParams())
            
            # Store current task for cleanup
            self.current_task = task
            
            # Run the pipeline
            runner = PipelineRunner()
            await runner.run(task)
            
        except Exception as e:
            logger.error(f"Voice session error: {e}")
            raise

# Global voice bot instance
voice_bot = VoiceBot()

# Web server for HTTP endpoints
async def create_room(request):
    """Create a new Daily.co room for voice chat"""
    try:
        # Create temporary room using Daily API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.daily.co/v1/rooms",
                headers={
                    "Authorization": f"Bearer {DAILY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "properties": {
                        "max_participants": 2,
                        "enable_chat": False,
                        "enable_screenshare": False,
                        "enable_recording": False,
                        "exp": int(asyncio.get_event_loop().time()) + 3600,  # 1 hour expiry
                    }
                }
            )
            
            if response.status_code == 201:
                room_data = response.json()
                room_url = room_data["url"]
                
                # Start voice session in background
                asyncio.create_task(voice_bot.start_voice_session(room_url))
                
                return web.json_response({
                    "room_url": room_url,
                    "status": "created",
                    "session_id": voice_bot.n8n_processor.session_id
                })
            else:
                logger.error(f"Failed to create room: {response.status_code}")
                return web.json_response(
                    {"error": "Failed to create room"}, 
                    status=500
                )
                
    except Exception as e:
        logger.error(f"Room creation error: {e}")
        return web.json_response(
            {"error": str(e)}, 
            status=500
        )

async def health_check(request):
    """Health check endpoint for Railway"""
    return web.json_response({
        "status": "healthy",
        "service": "pipecat-voice-ai",
        "timestamp": str(asyncio.get_event_loop().time()),
        "environment": {
            "deepgram": "configured" if DEEPGRAM_API_KEY else "missing",
            "elevenlabs": "configured" if ELEVENLABS_API_KEY else "missing", 
            "n8n_webhook": "configured" if N8N_WEBHOOK_URL else "missing",
            "daily": "configured" if DAILY_API_KEY else "missing"
        }
    })

async def voice_widget(request):
    """Serve the voice widget HTML"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipecat Voice AI</title>
    <script src="https://unpkg.com/@daily-co/daily-js"></script>
    <style>
        .voice-widget {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .voice-button {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            color: white;
            font-size: 24px;
        }
        .voice-button:hover {
            transform: scale(1.05);
        }
        .voice-button.active {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .status-display {
            position: absolute;
            bottom: 85px;
            right: 0;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            opacity: 0;
            transition: opacity 0.3s ease;
            min-width: 200px;
            text-align: center;
        }
        .status-display.visible {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="voice-widget">
        <button class="voice-button" id="voiceButton">🎤</button>
        <div class="status-display" id="statusDisplay">Click to start voice chat</div>
    </div>

    <script>
        class PipecatVoiceWidget {
            constructor() {
                this.callFrame = null;
                this.isActive = false;
                this.button = document.getElementById('voiceButton');
                this.status = document.getElementById('statusDisplay');
                
                this.button.addEventListener('click', () => this.toggleVoice());
            }
            
            async toggleVoice() {
                if (!this.isActive) {
                    await this.startVoiceChat();
                } else {
                    await this.stopVoiceChat();
                }
            }
            
            async startVoiceChat() {
                try {
                    this.updateStatus('Connecting...');
                    
                    // Create Daily.co room
                    const response = await fetch('/create-room', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.error || 'Failed to create room');
                    }
                    
                    // Join the room
                    this.callFrame = DailyIframe.createFrame({
                        showLeaveButton: false,
                        showFullscreenButton: false,
                        showLocalVideo: false,
                        showParticipantsBar: false,
                        theme: {
                            colors: {
                                accent: '#667eea',
                                accentText: '#FFFFFF'
                            }
                        }
                    });
                    
                    await this.callFrame.join({
                        url: data.room_url,
                        userName: 'User'
                    });
                    
                    // Set up event listeners
                    this.callFrame.on('participant-joined', () => {
                        this.updateStatus('🎙️ Voice AI ready - speak naturally');
                        this.button.classList.add('active');
                        this.button.textContent = '🔴';
                        this.isActive = true;
                    });
                    
                    this.callFrame.on('participant-left', () => {
                        this.stopVoiceChat();
                    });
                    
                    this.callFrame.on('error', (error) => {
                        console.error('Call error:', error);
                        this.updateStatus('Connection error - please try again');
                        this.stopVoiceChat();
                    });
                    
                } catch (error) {
                    console.error('Failed to start voice chat:', error);
                    this.updateStatus('Failed to connect: ' + error.message);
                }
            }
            
            async stopVoiceChat() {
                if (this.callFrame) {
                    await this.callFrame.leave();
                    this.callFrame.destroy();
                    this.callFrame = null;
                }
                
                this.isActive = false;
                this.button.classList.remove('active');
                this.button.textContent = '🎤';
                this.updateStatus('Click to start voice chat');
            }
            
            updateStatus(message) {
                this.status.textContent = message;
                this.status.classList.add('visible');
                
                setTimeout(() => {
                    this.status.classList.remove('visible');
                }, 3000);
            }
        }
        
        // Initialize widget
        document.addEventListener('DOMContentLoaded', () => {
            new PipecatVoiceWidget();
        });
    </script>
</body>
</html>
    """
    return web.Response(text=html_content, content_type='text/html')

async def init_app():
    """Initialize the web application"""
    app = web.Application()
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get('/health', health_check)
    app.router.add_get('/', voice_widget)
    app.router.add_post('/create-room', create_room)
    
    # Enable CORS on all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    """Main application entry point"""
    # Validate environment variables
    missing_vars = []
    if not DEEPGRAM_API_KEY:
        missing_vars.append("DEEPGRAM_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing_vars.append("ELEVENLABS_API_KEY")
    if not N8N_WEBHOOK_URL:
        missing_vars.append("N8N_WEBHOOK_URL")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Start web server
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"🎙️ Pipecat Voice AI Service running on port {PORT}")
    logger.info(f"🔗 Health check: http://localhost:{PORT}/health")
    logger.info(f"🎤 Voice widget: http://localhost:{PORT}/")
    
    # Keep the server running
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
