#!/usr/bin/env python3

import asyncio
import aiohttp
import os
import logging
from typing import Optional

# Pipecat imports
from pipecat.frames.frames import (
    Frame, AudioRawFrame, TextFrame, TranscriptionFrame, 
    TTSAudioRawFrame, LLMMessagesFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

# Environment setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") 
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
DAILY_API_KEY = os.getenv("DAILY_API_KEY")  # We'll need this for WebRTC

class N8NLLMService:
    """Custom LLM service that calls your N8N workflow instead of OpenAI"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session_id = None
        
    async def process_frame(self, frame: Frame) -> Optional[Frame]:
        if isinstance(frame, TranscriptionFrame):
            # User spoke - send to N8N
            user_message = frame.text
            logger.info(f"User said: {user_message}")
            
            try:
                response = await self.call_n8n(user_message)
                logger.info(f"N8N response: {response[:100]}...")
                
                # Return text frame for TTS
                return TextFrame(text=response)
                
            except Exception as e:
                logger.error(f"N8N error: {e}")
                return TextFrame(text="I'm sorry, I'm having trouble connecting right now.")
        
        return frame
    
    async def call_n8n(self, message: str) -> str:
        """Call your N8N workflow"""
        if not self.session_id:
            self.session_id = f"pipecat_{asyncio.get_event_loop().time()}"
            
        payload = {
            "message": message,
            "sessionId": self.session_id,
            "channel": "voice_call_realtime"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("output", "I didn't understand that.")
                else:
                    raise Exception(f"HTTP {response.status}")

class RealTimeVoiceBot:
    def __init__(self):
        self.transport = None
        self.pipeline = None
        
    async def create_room(self) -> str:
        """Create a Daily.co room for WebRTC"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.daily.co/v1/rooms",
                headers={
                    "Authorization": f"Bearer {DAILY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "properties": {
                        "max_participants": 2,
                        "enable_chat": False,
                        "enable_knocking": False,
                        "enable_screenshare": False,
                        "enable_recording": False,
                        "start_audio_off": False,
                        "start_video_off": True,
                        "exp": int(asyncio.get_event_loop().time()) + 3600  # 1 hour
                    }
                }
            ) as response:
                if response.status == 200:
                    room_data = await response.json()
                    return room_data["url"]
                else:
                    raise Exception(f"Failed to create room: {response.status}")
    
    async def setup_pipeline(self, room_url: str):
        """Set up the real-time voice processing pipeline"""
        
        # Transport (WebRTC via Daily.co)
        transport = DailyTransport(
            room_url=room_url,
            token=None,  # We'll join as guest
            bot_name="Voice AI Assistant",
            params=DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                video_out_enabled=False,
                video_in_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True
            )
        )
        
        # Services
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            model="nova-2",
            language="en-US",
            interim_results=True,
            smart_format=True,
            sample_rate=16000,
            channels=1
        )
        
        # Your N8N LLM service
        llm = N8NLLMService(webhook_url=N8N_WEBHOOK_URL)
        
        # TTS Service (using ElevenLabs via custom wrapper)
        tts = await self.create_elevenlabs_tts()
        
        # Context for conversation memory
        context = OpenAILLMContext()
        
        # Pipeline: Audio Input -> STT -> N8N -> TTS -> Audio Output
        pipeline = Pipeline([
            transport.input(),           # WebRTC audio input
            stt,                        # Speech to text (Deepgram)
            llm,                        # Your N8N workflow
            tts,                        # Text to speech (ElevenLabs)
            transport.output(),         # WebRTC audio output
            context
        ])
        
        self.transport = transport
        self.pipeline = pipeline
        
        return transport, pipeline
    
    async def create_elevenlabs_tts(self):
        """Create ElevenLabs TTS service"""
        # For now, we'll use a simple TTS wrapper
        # In production, you'd want streaming TTS
        class ElevenLabsTTS:
            def __init__(self):
                self.api_key = ELEVENLABS_API_KEY
                self.voice_id = ELEVENLABS_VOICE_ID
                
            async def process_frame(self, frame: Frame) -> Optional[Frame]:
                if isinstance(frame, TextFrame):
                    # Convert text to speech
                    audio_data = await self.text_to_speech(frame.text)
                    return TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=24000,
                        num_channels=1
                    )
                return frame
            
            async def text_to_speech(self, text: str) -> bytes:
                """Call ElevenLabs TTS API"""
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
                
                payload = {
                    "text": text,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                }
                
                headers = {
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            return await response.read()
                        else:
                            logger.error(f"TTS error: {response.status}")
                            return b""  # Return empty audio on error
        
        return ElevenLabsTTS()
    
    async def run(self):
        """Start the real-time voice bot"""
        try:
            # Create WebRTC room
            logger.info("Creating WebRTC room...")
            room_url = await self.create_room()
            logger.info(f"Room created: {room_url}")
            
            # Setup pipeline
            logger.info("Setting up real-time pipeline...")
            transport, pipeline = await self.setup_pipeline(room_url)
            
            # Start the pipeline
            logger.info("Starting real-time voice bot...")
            task = PipelineTask(pipeline)
            runner = PipelineRunner()
            
            await runner.run(task)
            
        except Exception as e:
            logger.error(f"Error running voice bot: {e}")
            raise

async def main():
    """Main entry point"""
    logger.info("🎙️ Starting Real-Time Voice AI Bot")
    
    # Validate environment
    required_vars = [
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY", 
        "ELEVENLABS_VOICE_ID",
        "N8N_WEBHOOK_URL",
        "DAILY_API_KEY"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        return
    
    # Start the bot
    bot = RealTimeVoiceBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
