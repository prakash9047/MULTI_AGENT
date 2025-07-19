import dotenv
dotenv.load_dotenv(dotenv_path="config/secrets.env")
from typing import Dict, Optional
import numpy as np

# Handle optional dependencies with fallbacks
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Speech recording will be unavailable.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not available. Using fallback for speech recognition.")

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: TTS not available. Using fallback for text-to-speech.")
import tempfile
import os
from langchain.agents import Tool
from config.settings import settings
from agents.agent_helpers import SimpleAgent

class VoiceAgent:
    def __init__(self):
        self.stt_model = self._initialize_stt()
        self.tts_model = self._initialize_tts()
        self.agent = self._create_agent()
        self.sample_rate = 16000
        
    def _initialize_stt(self):
        """Initialize the speech-to-text model."""
        if not WHISPER_AVAILABLE:
            print("Using mock STT model since whisper is not available")
            return None
            
        try:
            return whisper.load_model(settings.WHISPER_MODEL)
        except Exception as e:
            print(f"Error initializing STT model: {str(e)}")
            return None
    
    def _initialize_tts(self):
        """Initialize the text-to-speech model."""
        if not TTS_AVAILABLE:
            print("Using mock TTS model since TTS is not available")
            return None
            
        try:
            return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        except Exception as e:
            print(f"Error initializing TTS model: {str(e)}")
            return None
    
    def _create_agent(self):
        """Create the agent for voice operations."""
        return SimpleAgent(
            role="Voice Interface",
            goal="Handle speech-to-text and text-to-speech operations",
            backstory="Expert in voice processing and natural language interaction",
            tools=[
                Tool(
                    name="speech_to_text",
                    func=self.speech_to_text,
                    description="Convert speech to text"
                ),
                Tool(
                    name="text_to_speech",
                    func=self.text_to_speech,
                    description="Convert text to speech"
                )
            ]
        )
    
    def speech_to_text(self, audio_data: np.ndarray = None, mock: bool = True) -> Dict:
        """Convert speech to text."""
        try:
            # Use mock data for development if requested or if whisper is not available
            if mock or not WHISPER_AVAILABLE:
                print("Using mock speech-to-text data for development")
                return {
                    "text": "Tell me about recent technology stocks performance",
                    "confidence": 0.95,
                    "note": "Using mock STT data"
                }
                
            if audio_data is None:
                # Record audio if not provided
                if not SOUNDDEVICE_AVAILABLE:
                    print("Recording not available (sounddevice missing)")
                    return {
                        "text": "Tell me about recent market trends",
                        "confidence": 0.8,
                        "note": "Using default text due to missing sounddevice module"
                    }
                    
                try:
                    audio_data = self._record_audio()
                except Exception as audio_error:
                    print(f"Audio recording error: {str(audio_error)}")
                    return {
                        "text": "Tell me about recent market trends",
                        "confidence": 0.8,
                        "note": "Using default text due to recording error"
                    }
            
            if self.stt_model and len(audio_data) > 0:
                # Transcribe audio
                result = self.stt_model.transcribe(audio_data)
                return {
                    "text": result["text"],
                    "confidence": result.get("confidence", 0.0)
                }
            else:
                print("STT model not initialized or empty audio data")
                return {
                    "text": "Tell me about recent market trends",
                    "confidence": 0.8,
                    "note": "Using default text due to model or audio issue"
                }
        except Exception as e:
            print(f"STT error: {str(e)}")
            return {
                "text": "Tell me about recent market trends",
                "confidence": 0.8,
                "note": f"Using default text due to error: {str(e)}"
            }
                
    def text_to_speech(self, text: str, max_text_length: int = 1000) -> Dict:
        """Convert text to speech."""
        try:
            # If TTS is not available, return mock data
            if not TTS_AVAILABLE:
                print("Using mock text-to-speech data (TTS module not available)")
                return {
                    "audio_file": "mock_audio.wav",
                    "text": text,
                    "note": "Using mock audio file due to missing TTS module"
                }
                
            # Limit text length to avoid TTS issues
            if len(text) > max_text_length:
                print(f"Text too long ({len(text)} chars), truncating to {max_text_length} chars")
                text = text[:max_text_length] + "... (text truncated)"
            
            if self.tts_model:
                try:
                    # Generate speech
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        self.tts_model.tts_to_file(
                            text=text,
                            file_path=temp_file.name
                        )
                        return {
                            "audio_file": temp_file.name,
                            "text": text
                        }
                except Exception as tts_error:
                    print(f"TTS generation error: {str(tts_error)}")
                    # Return a mock audio file path for development/testing
                    return {
                        "audio_file": "mock_audio.wav",
                        "text": text,
                        "note": "Using mock audio file due to TTS error"
                    }
            else:
                print("TTS model not initialized")
                return {
                    "audio_file": "mock_audio.wav",
                    "text": text,
                    "note": "Using mock audio file due to uninitialized TTS model"
                }
        except Exception as e:
            print(f"TTS error: {str(e)}")
            return {
                "audio_file": "mock_audio.wav",
                "text": text,
                "note": f"Using mock audio file due to error: {str(e)}"
            }
    
    def _record_audio(self, duration: int = 5) -> np.ndarray:
        """Record audio from microphone."""
        if not SOUNDDEVICE_AVAILABLE:
            print("Cannot record audio: sounddevice module not available")
            return np.array([])
            
        try:
            print(f"Recording for {duration} seconds...")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            return audio_data.flatten()
        except Exception as e:
            print(f"Error recording audio: {str(e)}")
            return np.array([])
    
    def play_audio(self, audio_file: str):
        """Play audio from file."""
        if not SOUNDDEVICE_AVAILABLE:
            print("Cannot play audio: sounddevice module not available")
            return {"error": "sounddevice module not available", "status": "mock"}
            
        try:
            if os.path.exists(audio_file):
                data, samplerate = sd.read(audio_file)
                sd.play(data, samplerate)
                sd.wait()
                return {"status": "success"}
            else:
                return {"error": "Audio file not found"}
        except Exception as e:
            return {"error": str(e)}

    def process_voice_query(self, query: str = None) -> Dict:
        """Process a voice query and return the response."""
        try:
            # Convert speech to text if query is not provided
            if query is None:
                stt_result = self.speech_to_text()
                if "error" in stt_result:
                    return stt_result
                query = stt_result["text"]
            
            # Generate response (to be implemented by other agents)
            response = f"Processed query: {query}"
            
            # Convert response to speech
            tts_result = self.text_to_speech(response)
            if "error" in tts_result:
                return tts_result
            
            # Play the response
            play_result = self.play_audio(tts_result["audio_file"])
            
            return {
                "query": query,
                "response": response,
                "audio_file": tts_result["audio_file"],
                "play_status": play_result
            }
        except Exception as e:
            return {"error": str(e)} 