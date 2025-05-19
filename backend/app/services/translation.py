from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        # Initialize Google Cloud clients
        self.translate_client = translate.Client()
        self.tts_client = texttospeech.TextToSpeechClient()
        
        # Create directory for audio files
        self.audio_dir = Path(__file__).parent.parent / "static" / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    def translate_text(self, text: str, target_language: str) -> dict:
        """
        Translate text to target language
        """
        try:
            result = self.translate_client.translate(
                text,
                target_language=target_language
            )
            
            return {
                "translated_text": result["translatedText"],
                "source_language": result["detectedSourceLanguage"],
                "target_language": target_language
            }
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise
    
    def text_to_speech(self, text: str, language: str = "en") -> str:
        """
        Convert text to speech and return audio file path
        """
        try:
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=language,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )

            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            # Perform the text-to-speech request
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            # Generate unique filename
            filename = f"{hash(text)}_{language}.mp3"
            audio_path = self.audio_dir / filename

            # Write the response to the output file
            with open(audio_path, "wb") as out:
                out.write(response.audio_content)

            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            raise
