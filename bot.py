#!/usr/bin/env python3
"""
Pipecat Cloud Voice Bot with N8N Integration
This replaces your entire main.py + pipecat_bot.py with just this simple file!
"""

import os
import httpx
import logging
import time
from typing import Dict, Any
from pipecat.cloud import BaseBot

# Set up logging
logger = logging.getLogger(__name__)


class N8NVoiceBot(BaseBot):
    """
    Your voice bot that connects to N8N for AI responses.
    Pipecat Cloud handles ALL the complex stuff - we just focus on conversation!
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Your N8N webhook URL
        self.n8n_webhook_url = os.getenv("N8N_WEBHOOK_URL")
        
        # Session tracking (like your old code)
        self.session_id = f"voice_{int(time.time())}"
        
        logger.info(f"🤖 N8N Voice Bot initialized with session: {self.session_id}")
    
    async def on_start(self):
        """Called when the bot starts - perfect for welcome messages!"""
        logger.info("🎙️ Voice session started")
        
        # Optional: Send a welcome message
        await self.say("Hi! I'm your AI assistant. How can I help you today?")
    
    async def on_user_message(self, text: str) -> None:
        """
        This is called whenever the user speaks!
        Pipecat Cloud already did the speech-to-text for you.
        """
        
        # Skip very short inputs (like "um", "uh")
        if len(text.strip()) < 3:
            logger.debug(f"Ignoring short input: '{text}'")
            return
        
        logger.info(f"🎤 User said: '{text}'")
        
        try:
            # Call your N8N webhook
            ai_response = await self._call_n8n_webhook(text)
            
            # Speak the response (Pipecat Cloud handles text-to-speech!)
            await self.say(ai_response)
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            await self.say("I'm having some technical difficulties. Please try again.")
    
    async def _call_n8n_webhook(self, user_message: str) -> str:
        """
        Call your N8N AI Agent workflow.
        This is the EXACT same format your current webhook expects!
        """
        
        # Prepare the payload (matches your N8N Edit Fields node)
        payload = {
            "message": user_message,
            "sessionId": self.session_id,
            "channel": "voice_call"
        }
        
        logger.info(f"📤 Calling N8N webhook: {payload}")
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.post(
                    self.n8n_webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"📥 N8N response received")
                
                # Extract the AI response (adjust based on your N8N output)
                # Your webhook returns the AI agent's response
                ai_response = result.get("output", result.get("text", ""))
                
                if not ai_response:
                    logger.warning("Empty response from N8N")
                    return "I didn't get a proper response. Could you repeat that?"
                
                return ai_response
                
            except httpx.TimeoutException:
                logger.error("⏰ N8N request timed out")
                return "The response is taking too long. Let me try again."
                
            except httpx.HTTPStatusError as e:
                logger.error(f"🔥 N8N HTTP error: {e.response.status_code}")
                return "I'm having trouble connecting to my knowledge base."
                
            except Exception as e:
                logger.error(f"💥 Unexpected N8N error: {e}")
                return "Something went wrong. Please try again."
    
    async def on_end(self):
        """Called when the conversation ends"""
        logger.info(f"👋 Voice session ended. Session ID: {self.session_id}")
    
    async def on_error(self, error: Exception):
        """Handle any errors gracefully"""
        logger.error(f"🚨 Bot error: {error}")
        await self.say("I encountered an error. Please try again.")


# Pipecat Cloud automatically creates and runs your bot!
# No need for complex initialization - just define the class.
bot = N8NVoiceBot()
