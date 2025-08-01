import os
import json
import asyncio
import time
from typing import Optional
import httpx
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyTransport, DailyParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger


class N8NProcessor(FrameProcessor):
    """Custom processor to handle N8N webhook integration"""
    
    def __init__(self, webhook_url: str, session_id: str):
        super().__init__()
        self.webhook_url = webhook_url
        self.session_id = session_id
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_frame(self, frame, direction: FrameDirection):
        """Process frames and send user messages to N8N"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            # Extract user message
            for msg in frame.messages:
                if msg["role"] == "user":
                    await self._send_to_n8n(msg["content"])
    
    async def _send_to_n8n(self, message: str):
        """Send message to N8N webhook and get response"""
        try:
            payload = {
                "message": message,
                "sessionId": self.session_id,
                "channel": "voice"
            }
            
            logger.info(f"Sending to N8N: {message}")
            response = await self.client.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            ai_response = data.get("response", "I didn't understand that. Could you please repeat?")
            
            # Push AI response as a new frame
            await self.push_frame(LLMMessagesFrame([
                {"role": "assistant", "content": ai_response}
            ]))
            
        except Exception as e:
            logger.error(f"N8N webhook error: {e}")
            await self.push_frame(LLMMessagesFrame([
                {"role": "assistant", "content": "I'm having trouble connecting to my brain. Please try again."}
            ]))
    
    async def cleanup(self):
        """Cleanup HTTP client"""
        await self.client.aclose()


class VoiceBot:
    """Main voice bot class"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.room_url: Optional[str] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self.transport: Optional[DailyTransport] = None
        self.start_time = time.time()
        self.last_activity = time.time()
        
        # Load configuration
        self.n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
        self.daily_api_key = os.getenv("DAILY_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    
    async def create_room(self) -> str:
        """Create a Daily room for the session"""
        self.transport = DailyTransport(
            self.session_id,
            DailyParams(
                api_key=self.daily_api_key,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )
        
        # Create room and get URL
        room = await self.transport.create_room()
        self.room_url = room.url
        return self.room_url
    
    async def start(self):
        """Start the voice bot pipeline"""
        try:
            # Initialize STT service (Deepgram)
            stt = DeepgramSTTService(
                api_key=self.deepgram_api_key,
                params={
                    "model": "nova-2-general",
                    "language": "en",
                    "encoding": "linear16",
                    "sample_rate": 16000,
                    "channels": 1,
                    "interim_results": True,
                    "endpointing": 100,  # Faster end-of-speech detection
                }
            )
            
            # Initialize TTS service (ElevenLabs)
            tts = ElevenLabsTTSService(
                api_key=self.elevenlabs_api_key,
                voice_id=self.elevenlabs_voice_id,
                params={
                    "optimize_streaming_latency": 2,  # Lowest latency
                    "output_format": "pcm_16000",
                }
            )
            
            # Initialize N8N processor
            n8n_processor = N8NProcessor(self.n8n_webhook_url, self.session_id)
            
            # Build pipeline
            pipeline = Pipeline([
                self.transport.input(),    # Audio input from Daily
                stt,                       # Speech to text
                LLMUserResponseAggregator(),  # Aggregate user speech
                n8n_processor,             # Process with N8N
                tts,                       # Text to speech
                self.transport.output()    # Audio output to Daily
            ])
            
            # Set up runner
            self.runner = PipelineRunner()
            
            # Run pipeline
            self.task = self.runner.run(pipeline)
            
            # Join Daily room
            await self.transport.join()
            
            # Send welcome message
            await n8n_processor.push_frame(LLMMessagesFrame([
                {"role": "assistant", "content": "Hello! I'm your N8N voice assistant. How can I help you today?"}
            ]))
            
            logger.info(f"Voice bot started for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error starting voice bot: {e}")
            raise
    
    async def stop(self):
        """Stop the voice bot"""
        try:
            if self.task:
                self.task.cancel()
            
            if self.transport:
                await self.transport.leave()
                await self.transport.cleanup()
            
            logger.info(f"Voice bot stopped for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error stopping voice bot: {e}")
    
    async def is_inactive(self) -> bool:
        """Check if session has been inactive for too long"""
        # Consider inactive after 5 minutes without activity
        return (time.time() - self.last_activity) > 300
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
