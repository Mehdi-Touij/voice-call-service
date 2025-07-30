#!/usr/bin/env python3

import os
import asyncio
import websockets
import json
import aiohttp
import base64
from datetime import datetime

# Environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") 
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8080))

class VoiceProcessor:
    def __init__(self):
        self.active_calls = {}
        
    async def speech_to_text(self, audio_data):
        """Convert speech to text using Deepgram"""
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        params = {
            "model": "nova-3",
            "language": "en",
            "punctuate": "true",
            "smart_format": "true"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=audio_data, params=params) as response:
                result = await response.json()
                transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
                return transcript
    
    async def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                audio_data = await response.read()
                return audio_data
    
    async def get_ai_response(self, message, call_id):
        """Get AI response from N8N workflow"""
        payload = {
            "message": message,
            "sessionId": f"voice_{call_id}",
            "channel": "voice_call",
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(N8N_WEBHOOK_URL, json=payload) as response:
                result = await response.json()
                return result.get("output", "I'm sorry, I didn't understand that.")
    
    async def process_voice_call(self, websocket, path):
        """Handle incoming voice call via WebSocket"""
        call_id = f"call_{datetime.now().timestamp()}"
        self.active_calls[call_id] = {"websocket": websocket, "status": "active"}
        
        print(f"New voice call started: {call_id}")
        
        try:
            await websocket.send(json.dumps({
                "type": "call_started",
                "call_id": call_id,
                "message": "Voice call connected. Start speaking!"
            }))
            
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "audio":
                    # Decode base64 audio
                    audio_data = base64.b64decode(data["audio"])
                    
                    # Speech to text
                    transcript = await self.speech_to_text(audio_data)
                    print(f"Transcript: {transcript}")
                    
                    if transcript.strip():
                        # Get AI response
                        ai_response = await self.get_ai_response(transcript, call_id)
                        print(f"AI Response: {ai_response}")
                        
                        # Text to speech
                        audio_response = await self.text_to_speech(ai_response)
                        
                        # Send back to client
                        response_data = {
                            "type": "response",
                            "transcript": transcript,
                            "ai_response": ai_response,
                            "audio": base64.b64encode(audio_response).decode()
                        }
                        await websocket.send(json.dumps(response_data))
                
                elif data["type"] == "end_call":
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Call {call_id} disconnected")
        finally:
            if call_id in self.active_calls:
                del self.active_calls[call_id]

# Health check endpoint
async def health_check(request):
    return web.Response(text="Voice service is healthy")

# Start the voice processing service
async def main():
    processor = VoiceProcessor()
    
    # Start WebSocket server for voice calls
    print(f"Starting voice processing service on port {PORT}")
    print(f"WebSocket endpoint: ws://localhost:{PORT}/voice")
    
    server = await websockets.serve(
        processor.process_voice_call, 
        "0.0.0.0", 
        PORT
    )
    
    print("Voice processing service is running...")
    await server.wait_closed()

if __name__ == "__main__":
    # Check required environment variables
    required_vars = [
        "DEEPGRAM_API_KEY", 
        "ELEVENLABS_API_KEY", 
        "ELEVENLABS_VOICE_ID", 
        "N8N_WEBHOOK_URL"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"ERROR: Missing environment variables: {missing_vars}")
        exit(1)
    
    asyncio.run(main())
