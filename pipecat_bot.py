#!/usr/bin/env python3
"""
Pipecat voice bot implementation with N8N integration
"""

import logging
import time
from typing import AsyncGenerator

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

logger = logging.getLogger(__name__)


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
    
    def __init__(self, n8n_webhook_url: str, deepgram_api_key: str, elevenlabs_api_key: str, elevenlabs_voice_id: str):
        if not n8n_webhook_url:
            raise ValueError("N8N_WEBHOOK_URL is not configured")
        
        self.n8n_processor = N8NLLMProcessor(n8n_webhook_url)
        self.deepgram_api_key = deepgram_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
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
            api_key=self.deepgram_api_key,
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
            api_key=self.elevenlabs_api_key,
            voice_id=self.elevenlabs_voice_id,
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
