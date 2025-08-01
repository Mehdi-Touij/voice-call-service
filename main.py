#!/usr/bin/env python3

import asyncio
import logging
import os
import sys
import time
from typing import AsyncGenerator

# Fix potential import issues by explicitly importing what we need
from aiohttp import web
import aiohttp_cors
import json
import httpx

# Pipecat core imports
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

# Configure logging with explicit handler to avoid potential issues
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Environment variables with validation
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") 
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "aD6riP1btT197c6dACmy")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
# Daily domain from Pipecat Cloud (not an API key)
DAILY_DOMAIN = os.getenv("DAILY_DOMAIN", "cloud-5bbf6640b3fd4778867e168bfbe3ca06")
PORT = int(os.getenv("PORT", "8080"))

# Validate environment variables at startup
if not all([DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, N8N_WEBHOOK_URL]):
    logger.warning("Missing required environment variables. Some features may not work.")

class N8NLLMProcessor(FrameProcessor):
    """
    Pipecat processor that integrates with your N8N AI Agent workflow
    Handles the exact payload format your N8N webhook expects
    """
    
    def __init__(self, webhook_url: str):
        super().__init__()
        self.webhook_url = webhook_url
        self.session_id = f"voice_{int(time.time())}"
        logger.info(f"N8N Processor initialized with session: {self.session_id}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> AsyncGenerator[Frame, None]:
        """Process frames through the Pipecat pipeline"""
        
        if isinstance(frame, TranscriptionFrame):
            # Extract user speech
            user_text = frame.text.strip()
            
            if user_text and len(user_text) > 2:  # Ignore very short transcriptions
                logger.info(f"🎤 User said: '{user_text}'")
                
                try:
                    # Send to your N8N AI Agent
                    ai_response = await self.call_n8n_agent(user_text)
                    logger.info(f"🤖 AI responded: '{ai_response}'")
                    
                    # Send AI response for TTS
                    yield TextFrame(ai_response)
                    
                except Exception as e:
                    logger.error(f"❌ N8N processing failed: {e}")
                    error_response = "I'm having some technical difficulties. Please try again."
                    yield TextFrame(error_response)
            else:
                logger.debug(f"Ignoring short transcription: '{user_text}'")
        else:
            # Pass through other frame types
            yield frame
    
    async def call_n8n_agent(self, text: str) -> str:
        """
        Call your N8N AI Agent with the exact payload format it expects
        Based on your workflow: message, sessionId, channel
        """
        try:
            # Payload format matching your N8N "Edit Fields" node
            payload = {
                "message": text,
                "sessionId": self.session_id, 
                "channel": "voice_call"
            }
            
            logger.info(f"📤 Sending to N8N: {payload}")
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"📥 N8N response: {result}")
                
                # Extract AI response from your N8N workflow
                # Your "Respond to Webhook" node should return the AI Agent output
                ai_response = result.get("output", result.get("text", result.get("message", "I didn't understand that.")))
                
                return ai_response
                
        except httpx.TimeoutException:
            logger.error("⏰ N8N request timed out")
            raise Exception("AI processing timed out")
        except httpx.HTTPStatusError as e:
            logger.error(f"🔥 N8N HTTP error: {e.response.status_code}")
            raise Exception("AI service temporarily unavailable") 
        except Exception as e:
            logger.error(f"💥 Unexpected N8N error: {e}")
            raise

class VoiceBot:
    """Complete Pipecat voice bot with N8N integration"""
    
    def __init__(self):
        if not N8N_WEBHOOK_URL:
            raise ValueError("N8N_WEBHOOK_URL is not configured")
        self.n8n_processor = N8NLLMProcessor(N8N_WEBHOOK_URL)
        self.current_task = None
        self.is_running = False
        
    async def create_pipeline(self, transport):
        """Create the complete Pipecat voice processing pipeline"""
        
        logger.info("🔧 Building Pipecat pipeline...")
        
        # 1. Voice Activity Detection - Silero VAD
        vad = SileroVADAnalyzer(
            confidence_threshold=0.6,
            min_volume=0.6
        )
        logger.info("✅ VAD (Silero) configured")
        
        # 2. Speech-to-Text - Deepgram Nova-2
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2",
            language="en-US",
            interim_results=True,
            smart_format=True,
            utterance_end_ms=1000,
            vad_events=True
        )
        logger.info("✅ STT (Deepgram) configured")
        
        # 3. Text-to-Speech - ElevenLabs Turbo v2
        tts = ElevenLabsTTSService(
            api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            model="eleven_turbo_v2",
            optimize_streaming_latency=3
        )
        logger.info("✅ TTS (ElevenLabs) configured")
        
        # 4. Build the pipeline
        pipeline = Pipeline([
            transport.input(),     # WebRTC audio input
            vad,                  # Voice activity detection
            stt,                  # Speech to text
            self.n8n_processor,   # Your N8N AI Agent
            tts,                  # Text to speech
            transport.output(),   # WebRTC audio output
        ])
        
        logger.info("🎯 Pipecat pipeline built successfully")
        return pipeline
    
    async def start_voice_session(self, room_url: str):
        """Start a Pipecat voice session"""
        try:
            if self.is_running:
                logger.warning("⚠️ Voice session already running")
                return
                
            self.is_running = True
            logger.info(f"🚀 Starting Pipecat voice session: {room_url}")
            
            # Configure Daily.co transport
            transport = DailyTransport(
                room_url,
                None,  # No token needed
                "Voice Assistant",
                DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    transcription_enabled=False,
                    vad_enabled=True,
                    camera_enabled=False
                )
            )
            logger.info("✅ Daily.co transport configured")
            
            # Create and run pipeline
            pipeline = await self.create_pipeline(transport)
            task = PipelineTask(
                pipeline,
                PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True
                )
            )
            
            self.current_task = task
            
            # Run the Pipecat pipeline
            runner = PipelineRunner()
            logger.info("🎙️ Starting Pipecat pipeline runner...")
            await runner.run(task)
            
        except Exception as e:
            logger.error(f"💥 Pipecat session failed: {e}", exc_info=True)
            raise
        finally:
            self.is_running = False
            logger.info("🔚 Pipecat session ended")

async def create_room(request):
    """Create Daily room using Pipecat Cloud domain directly"""
    try:
        logger.info("🏗️ Creating Daily room...")
        
        # Check if N8N webhook URL is set
        if not N8N_WEBHOOK_URL:
            logger.error("❌ N8N_WEBHOOK_URL not set")
            response_data = json.dumps({"error": "N8N webhook URL not configured"})
            return web.Response(text=response_data, status=500, content_type='application/json')
        
        # Generate a unique room name
        room_name = f"voice-{int(time.time())}-{os.urandom(4).hex()}"
        
        # Use your Pipecat Cloud Daily domain
        room_url = f"https://{DAILY_DOMAIN}.daily.co/{room_name}"
        
        logger.info(f"✅ Room created: {room_url}")
        
        # Create voice bot
        voice_bot = VoiceBot()
        
        # Start Pipecat session in background
        asyncio.create_task(voice_bot.start_voice_session(room_url))
        
        response_data = json.dumps({
            "room_url": room_url,
            "status": "created",
            "session_id": voice_bot.n8n_processor.session_id,
            "framework": "pipecat"
        })
        return web.Response(text=response_data, content_type='application/json')
        
    except Exception as e:
        logger.error(f"💥 Room creation error: {e}", exc_info=True)
        response_data = json.dumps({"error": str(e)})
        return web.Response(text=response_data, status=500, content_type='application/json')

async def health_check(request):
    """Health check endpoint"""
    environment_status = {
        "deepgram": "configured" if DEEPGRAM_API_KEY else "missing",
        "elevenlabs": "configured" if ELEVENLABS_API_KEY else "missing",
        "n8n_webhook": "configured" if N8N_WEBHOOK_URL else "missing", 
        "daily": f"domain: {DAILY_DOMAIN}" if DAILY_DOMAIN else "missing"
    }
    
    response_data = json.dumps({
        "status": "healthy",
        "service": "pipecat-voice-ai-n8n",
        "framework": "pipecat",
        "version": "0.0.40",
        "timestamp": int(time.time()),
        "environment": environment_status,
        "features": {
            "vad": "silero",
            "stt": "deepgram_nova2", 
            "llm": "n8n_claude_haiku",
            "tts": "elevenlabs_turbo_v2"
        },
        "n8n_integration": {
            "workflow": "AI Agent + Memory + Knowledge Base",
            "session_id": "created_per_voice_session"
        }
    })
    
    return web.Response(text=response_data, content_type='application/json')

async def voice_widget(request):
    """Serve the Pipecat voice widget"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipecat Voice AI + N8N</title>
    <script src="https://cdn.daily.co/daily-js/daily.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .framework-badge {
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            display: inline-block;
            margin: 10px 0;
            backdrop-filter: blur(10px);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .status-card {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        .status-card:hover {
            transform: translateY(-5px);
        }
        .status-card h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 1.1rem;
        }
        .status-indicator {
            font-size: 2rem;
            margin-bottom: 15px;
        }
        .status-text {
            font-size: 14px;
            color: #666;
        }
        .voice-widget {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 10000;
        }
        .voice-button {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: white;
            font-size: 32px;
            position: relative;
        }
        .voice-button:hover {
            transform: scale(1.05) translateY(-2px);
            box-shadow: 0 16px 50px rgba(102, 126, 234, 0.5);
        }
        .voice-button.connecting {
            background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
            animation: connecting 1s ease-in-out infinite;
        }
        .voice-button.active {
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
            animation: listening 2s ease-in-out infinite;
        }
        @keyframes connecting {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        @keyframes listening {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        .status-display {
            position: absolute;
            bottom: 95px;
            right: 0;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px 20px;
            border-radius: 15px;
            font-size: 14px;
            opacity: 0;
            transition: all 0.3s ease;
            min-width: 280px;
            text-align: center;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }
        .status-display.visible {
            opacity: 1;
        }
        .status-display.error {
            background: rgba(244, 67, 54, 0.9);
        }
        .call-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 9999;
            display: none;
            align-items: center;
            justify-content: center;
        }
        .call-container {
            width: 90%;
            max-width: 600px;
            height: 70%;
            max-height: 400px;
            background: #1a1a1a;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }
        .call-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .call-title {
            font-size: 16px;
            font-weight: 600;
        }
        .close-button {
            background: none;
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            padding: 5px 10px;
            border-radius: 5px;
            transition: background 0.2s ease;
        }
        .close-button:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        .call-frame {
            width: 100%;
            height: calc(100% - 60px);
        }
        .logs {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .logs h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        .log-container {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            line-height: 1.4;
        }
        .log-entry {
            margin: 5px 0;
        }
        .log-entry.info { color: #2196f3; }
        .log-entry.success { color: #4caf50; }
        .log-entry.error { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Pipecat Voice AI</h1>
            <div class="framework-badge">Powered by Pipecat + N8N + Claude</div>
            <p>Professional real-time voice conversation with AI memory & knowledge base</p>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <div class="status-indicator" id="deepgramStatus">⏳</div>
                <h3>Deepgram STT</h3>
                <div class="status-text" id="deepgramText">Checking...</div>
            </div>
            <div class="status-card">
                <div class="status-indicator" id="elevenlabsStatus">⏳</div>
                <h3>ElevenLabs TTS</h3>
                <div class="status-text" id="elevenlabsText">Checking...</div>
            </div>
            <div class="status-card">
                <div class="status-indicator" id="n8nStatus">⏳</div>
                <h3>N8N AI Agent</h3>
                <div class="status-text" id="n8nText">Checking...</div>
            </div>
            <div class="status-card">
                <div class="status-indicator" id="dailyStatus">⏳</div>
                <h3>Daily.co WebRTC</h3>
                <div class="status-text" id="dailyText">Checking...</div>
            </div>
        </div>

        <div class="logs">
            <h3>🔍 Pipecat Session Logs</h3>
            <div class="log-container" id="logContainer"></div>
        </div>
    </div>

    <div class="voice-widget">
        <button class="voice-button" id="voiceButton">🎤</button>
        <div class="status-display" id="statusDisplay">Ready for voice chat</div>
    </div>

    <div class="call-overlay" id="callOverlay">
        <div class="call-container">
            <div class="call-header">
                <div class="call-title">🎙️ Pipecat + N8N Voice Session</div>
                <button class="close-button" id="closeButton">×</button>
            </div>
            <div class="call-frame" id="callFrame"></div>
        </div>
    </div>

    <script>
        class PipecatN8NWidget {
            constructor() {
                this.callFrame = null;
                this.isActive = false;
                this.button = document.getElementById('voiceButton');
                this.status = document.getElementById('statusDisplay');
                this.overlay = document.getElementById('callOverlay');
                this.callContainer = document.getElementById('callFrame');
                this.closeButton = document.getElementById('closeButton');
                this.logContainer = document.getElementById('logContainer');
                
                // Verify all elements exist
                if (!this.callContainer) {
                    console.error('Call container element not found!');
                }
                
                this.button.addEventListener('click', () => this.toggleVoice());
                this.closeButton.addEventListener('click', () => this.endCall());
                
                this.checkServiceHealth();
            }
            
            log(message, type = 'info') {
                const timestamp = new Date().toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry ' + type;
                logEntry.textContent = '[' + timestamp + '] ' + message;
                this.logContainer.appendChild(logEntry);
                this.logContainer.scrollTop = this.logContainer.scrollHeight;
                console.log('[Pipecat+N8N] ' + message);
            }
            
            updateServiceStatus(service, status, text = '') {
                const indicator = document.getElementById(service + 'Status');
                const textEl = document.getElementById(service + 'Text');
                
                if (indicator && textEl) {
                    if (status === 'configured' || status.includes('domain:')) {
                        indicator.textContent = '✅';
                        textEl.textContent = text || 'Connected';
                        textEl.style.color = '#4caf50';
                    } else {
                        indicator.textContent = '❌';
                        textEl.textContent = text || 'Missing Config';
                        textEl.style.color = '#f44336';
                    }
                }
            }
            
            async checkServiceHealth() {
                try {
                    this.log('Checking Pipecat + N8N service health...', 'info');
                    const response = await fetch('/health');
                    const health = await response.json();
                    
                    this.log('Service: ' + health.service + ' (' + health.framework + ')', 'success');
                    this.log('N8N Integration: ' + health.n8n_integration.workflow, 'info');
                    
                    // Update status indicators
                    Object.entries(health.environment).forEach(([service, status]) => {
                        if (service === 'daily' && status.includes('domain:')) {
                            this.updateServiceStatus(service, status, 'Domain Ready');
                        } else {
                            this.updateServiceStatus(service, status);
                        }
                    });
                    
                    const missing = Object.entries(health.environment)
                        .filter(([key, value]) => value === 'missing')
                        .map(([key]) => key);
                    
                    if (missing.length === 0) {
                        this.log('✅ All services configured - ready for voice chat!', 'success');
                    } else {
                        this.log('❌ Missing: ' + missing.join(', '), 'error');
                    }
                } catch (error) {
                    this.log('❌ Health check failed: ' + error.message, 'error');
                }
            }
            
            async toggleVoice() {
                if (!this.isActive) {
                    await this.startVoiceChat();
                } else {
                    await this.endCall();
                }
            }
            
            async startVoiceChat() {
                try {
                    // Clean up any existing call frame first
                    if (this.callFrame) {
                        try {
                            await this.callFrame.leave();
                            this.callFrame.destroy();
                            this.callFrame = null;
                        } catch (e) {
                            console.error('Error cleaning up existing frame:', e);
                        }
                    }
                    
                    this.button.classList.add('connecting');
                    this.updateStatus('Starting Pipecat session...', false);
                    this.log('🚀 Starting Pipecat + N8N voice session...', 'info');
                    
                    // Create Daily.co room
                    const response = await fetch('/create-room', { method: 'POST' });
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || 'Failed to create room');
                    }
                    
                    this.log('Room created: ' + data.room_url, 'success');
                    this.log('Session ID: ' + data.session_id, 'info');
                    
                    // Log what's available
                    console.log('Daily.co check - window.Daily:', typeof window.Daily);
                    console.log('Daily.co check - window.DailyIframe:', typeof window.DailyIframe);
                    console.log('Daily.co check - window.daily:', typeof window.daily);
                    
                    // Check if Daily is available (CDN version uses window.Daily)
                    if (typeof window.Daily === 'undefined') {
                        throw new Error('Daily.co library not loaded. Please refresh the page.');
                    }
                    
                    // Show call overlay first
                    this.overlay.style.display = 'flex';
                    
                    // Small delay to ensure DOM is ready
                    await new Promise(resolve => setTimeout(resolve, 100));
                    
                    // Ensure container exists
                    if (!this.callContainer) {
                        throw new Error('Call container element not found');
                    }
                    
                    console.log('Call container:', this.callContainer);
                    console.log('Container dimensions:', this.callContainer.offsetWidth, 'x', this.callContainer.offsetHeight);
                    
                    try {
                        // Create Daily.co frame using the CDN API
                        console.log('Creating Daily call frame...');
                        this.callFrame = window.Daily.createFrame(this.callContainer, {
                            iframeStyle: {
                                width: '100%',
                                height: '100%',
                                border: '0'
                            },
                            showLeaveButton: true,
                            showFullscreenButton: false
                        });
                        console.log('Daily frame created successfully:', this.callFrame);
                    } catch (frameError) {
                        console.error('Error creating Daily frame:', frameError);
                        throw new Error('Failed to create video frame: ' + frameError.message);
                    }
                    });
                    
                    // Set up event listeners before joining
                    this.setupCallEventListeners();
                    
                    // Now join the room with error handling
                    try {
                        console.log('Joining room:', data.room_url);
                        await this.callFrame.join({
                            url: data.room_url,
                            userName: 'User'
                        });
                        console.log('Successfully joined room');
                    } catch (joinError) {
                        console.error('Error joining room:', joinError);
                        throw new Error('Failed to join room: ' + joinError.message);
                    }
                    
                } catch (error) {
                    console.error('Voice chat error:', error);
                    this.log('Failed to start session: ' + error.message, 'error');
                    this.updateStatus('Failed: ' + error.message, true);
                    this.button.classList.remove('connecting', 'active');
                    this.overlay.style.display = 'none';
                    this.button.textContent = '🎤';
                    
                    // Clean up on error
                    if (this.callFrame) {
                        try {
                            const state = this.callFrame.meetingState();
                            if (state && state !== 'left' && state !== 'error') {
                                this.callFrame.destroy();
                            }
                        } catch (e) {
                            console.error('Error destroying frame:', e);
                        }
                        this.callFrame = null;
                    }
                    
                    this.isActive = false;
                }
            }
            
            setupCallEventListeners() {
                this.callFrame.on('joined-meeting', () => {
                    this.log('✅ Joined Daily.co meeting', 'success');
                    this.button.classList.remove('connecting');
                    this.button.classList.add('active');
                    this.button.textContent = '🔴';
                    this.updateStatus('🎙️ Pipecat AI ready! Speak naturally.');
                    this.isActive = true;
                });
                
                this.callFrame.on('participant-joined', (event) => {
                    this.log('Participant joined: ' + event.participant.user_name, 'info');
                    if (event.participant.user_name === 'Voice Assistant') {
                        this.log('🤖 Pipecat AI connected with N8N!', 'success');
                        this.updateStatus('🤖 AI + N8N ready!');
                    }
                });
                
                this.callFrame.on('track-started', (event) => {
                    if (event.track.kind === 'audio' && event.participant.user_name === 'Voice Assistant') {
                        this.log('🔊 AI is speaking...', 'info');
                        this.updateStatus('🔊 AI responding...');
                    }
                });
                
                this.callFrame.on('track-stopped', (event) => {
                    if (event.track.kind === 'audio' && event.participant.user_name === 'Voice Assistant') {
                        this.log('🎙️ Ready for next input', 'info');
                        this.updateStatus('🎙️ Listening...');
                    }
                });
                
                this.callFrame.on('error', (error) => {
                    this.log('Call error: ' + (error.errorMsg || error), 'error');
                    this.updateStatus('Connection error', true);
                    this.endCall();
                });
                
                this.callFrame.on('left-meeting', () => {
                    this.log('Left meeting', 'info');
                    this.endCall();
                });
            }
            
            async endCall() {
                try {
                    this.log('🔚 Ending Pipecat session...', 'info');
                    
                    if (this.callFrame) {
                        try {
                            // Check if we're in a meeting before trying to leave
                            const meetingState = this.callFrame.meetingState();
                            console.log('Meeting state:', meetingState);
                            
                            if (meetingState === 'joined' || meetingState === 'joining') {
                                await this.callFrame.leave();
                            }
                            
                            await this.callFrame.destroy();
                        } catch (e) {
                            console.error('Error during cleanup:', e);
                        }
                        this.callFrame = null;
                    }
                    
                    this.overlay.style.display = 'none';
                    this.isActive = false;
                    this.button.classList.remove('connecting', 'active');
                    this.button.textContent = '🎤';
                    this.updateStatus('Session ended');
                    this.log('Session ended successfully', 'success');
                    
                } catch (error) {
                    this.log('Error ending session: ' + error.message, 'error');
                    // Force cleanup even on error
                    this.callFrame = null;
                    this.overlay.style.display = 'none';
                    this.isActive = false;
                    this.button.classList.remove('connecting', 'active');
                    this.button.textContent = '🎤';
                }
            }
            
            updateStatus(message, isError = false) {
                this.status.textContent = message;
                this.status.classList.toggle('error', isError);
                this.status.classList.add('visible');
                
                if (!this.isActive) {
                    setTimeout(() => {
                        this.status.classList.remove('visible');
                    }, 4000);
                }
            }
        }
        
        // Initialize widget when everything is loaded
        window.addEventListener('load', () => {
            console.log('Window loaded, checking for Daily.co...');
            
            const initWidget = () => {
                if (typeof window.Daily !== 'undefined') {
                    console.log('Daily.co loaded successfully');
                    const widget = new PipecatN8NWidget();
                    window.pipecatWidget = widget;
                    console.log('🎙️ Pipecat + N8N Voice Widget ready!');
                } else {
                    console.log('Daily.co not yet loaded, waiting...');
                    setTimeout(initWidget, 500);
                }
            };
            
            initWidget();
        });
    </script>
</body>
</html>"""
    
    return web.Response(text=html_content, content_type='text/html')

def create_app():
    """Create web application with proper initialization"""
    app = web.Application()
    
    # CORS configuration
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Routes - use proper route registration
    app.router.add_get('/health', health_check)
    app.router.add_get('/', voice_widget)
    app.router.add_post('/create-room', create_room)
    
    # Enable CORS for all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    """Main application entry point"""
    logger.info("🎙️ Starting Bulletproof Pipecat + N8N Voice AI Service...")
    
    # Environment validation
    missing_vars = []
    if not DEEPGRAM_API_KEY:
        missing_vars.append("DEEPGRAM_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing_vars.append("ELEVENLABS_API_KEY")
    if not N8N_WEBHOOK_URL:
        missing_vars.append("N8N_WEBHOOK_URL")
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        logger.error("Set these in Railway dashboard before deployment")
    else:
        logger.info("✅ All required environment variables are set")
    
    logger.info(f"📍 Using Daily domain: {DAILY_DOMAIN}")
    
    # Create and start web server
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    
    logger.info(f"🚀 Pipecat Voice AI Service running on port {PORT}")
    logger.info(f"🔗 Health: http://localhost:{PORT}/health")
    logger.info(f"🎤 Widget: http://localhost:{PORT}/")
    logger.info(f"⚡ Features: Silero VAD + Deepgram STT + N8N Claude + ElevenLabs TTS")
    
    # Keep running
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("🔚 Shutting down...")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)
