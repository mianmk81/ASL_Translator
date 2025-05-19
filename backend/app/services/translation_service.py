from googletrans import Translator
from gtts import gTTS
import os
from typing import Dict

class TranslationService:
    def __init__(self):
        self.translator = Translator()
        
    async def translate_text(self, text: str, target_language: str) -> Dict[str, str]:
        """
        Translate text to target language
        """
        try:
            translation = self.translator.translate(text, dest=target_language)
            return {
                "original_text": text,
                "translated_text": translation.text,
                "source_language": translation.src,
                "target_language": target_language
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def text_to_speech(self, text: str, language: str = "en") -> Dict[str, str]:
        """
        Convert text to speech using gTTS
        """
        try:
            tts = gTTS(text=text, lang=language)
            # TODO: Implement proper file handling and storage
            audio_path = "temp.mp3"
            tts.save(audio_path)
            return {"audio_path": audio_path}
        except Exception as e:
            return {"error": str(e)}

# Initialize translation service
translation_service = TranslationService()
