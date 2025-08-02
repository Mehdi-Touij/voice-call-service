import os
import uuid
import asyncio
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pipecat_bot import VoiceBot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="N8N Voice Bot Service")

# Configure CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_sessions: Dict[str, VoiceBot] = {}

# Cleanup task reference
cleanup_task = None

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint for Digital Ocean"""
    return {"status": "healthy", "service": "voice-bot"}

@app.post("/start_session")
async def start_session():
    """Start a new voice chat session"""
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create new voice bot instance
        bot = VoiceBot(session_id)
        
        # Get Daily room URL
        room_url = await bot.create_room()
        
        # Store session
        active_sessions[session_id] = bot
        
        # Start bot in background
        asyncio.create_task(bot.start())
        
        return {
            "session_id": session_id,
            "room_url": room_url,
            "status": "ready"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_session/{session_id}")
async def end_session(session_id: str):
    """End an active voice chat session"""
    if session_id in active_sessions:
        bot = active_sessions[session_id]
        await bot.stop()
        del active_sessions[session_id]
        return {"status": "session_ended"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/session_status/{session_id}")
async def session_status(session_id: str):
    """Check the status of a session"""
    if session_id in active_sessions:
        return {"status": "active", "session_id": session_id}
    else:
        return {"status": "inactive", "session_id": session_id}

# Cleanup inactive sessions periodically
async def cleanup_sessions():
    """Remove inactive sessions after timeout"""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        inactive_sessions = []
        for session_id, bot in active_sessions.items():
            if await bot.is_inactive():
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            bot = active_sessions[session_id]
            await bot.stop()
            del active_sessions[session_id]
            print(f"Cleaned up inactive session: {session_id}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global cleanup_task
    print("🚀 Voice Bot Service Starting...")
    print(f"📡 N8N Webhook: {os.getenv('N8N_WEBHOOK_URL')}")
    print(f"🎙️  Deepgram: {'✓' if os.getenv('DEEPGRAM_API_KEY') else '✗'}")
    print(f"🔊 ElevenLabs: {'✓' if os.getenv('ELEVENLABS_API_KEY') else '✗'}")
    print(f"📹 Daily: {'✓' if os.getenv('DAILY_API_KEY') else '✗'}")
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_sessions())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global cleanup_task
    print("🛑 Shutting down Voice Bot Service...")
    
    # Cancel cleanup task
    if cleanup_task:
        cleanup_task.cancel()
    
    # End all active sessions
    for session_id, bot in active_sessions.items():
        await bot.stop()
    active_sessions.clear()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
