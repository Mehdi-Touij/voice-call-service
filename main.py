#!/usr/bin/env python3
"""
Main web server for Pipecat Voice AI with N8N integration
This file is now only 145 lines! The voice bot logic is in pipecat_bot.py
"""

import asyncio
import json
import logging
import os
import sys
import time

from aiohttp import web
import aiohttp_cors

from pipecat_bot import VoiceBot

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") 
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "aD6riP1btT197c6dACmy")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
DAILY_DOMAIN = os.getenv("DAILY_DOMAIN", "cloud-5bbf6640b3fd4778867e168bfbe3ca06")
PORT = int(os.getenv("PORT", "8080"))


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
        voice_bot = VoiceBot(
            n8n_webhook_url=N8N_WEBHOOK_URL,
            deepgram_api_key=DEEPGRAM_API_KEY,
            elevenlabs_api_key=ELEVENLABS_API_KEY,
            elevenlabs_voice_id=ELEVENLABS_VOICE_ID
        )
        
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
    try:
        # Read the HTML file from static directory
        static_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
        with open(static_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    except FileNotFoundError:
        logger.error("❌ static/index.html not found")
        return web.Response(text="Widget HTML not found", status=500)


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
    
    # Routes
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
