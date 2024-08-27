import os
from dataclasses import dataclass, field

@dataclass
class CoquiTTSHandlerArguments:
    coqui_model_name: str = field(
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        metadata={
            "help": "The name of the Coqui TTS model to use."
        },
    )
    coqui_speaker_wav: str = field(
        default="female.wav",
        metadata={
            "help": "Name of the WAV file for speaker conditioning in multi-speaker models. The file should be in the TTSM/audio folder."
        },
    )
    coqui_language: str = field(
        default="fr",
        metadata={
            "help": "The language of the text to be synthesized. Default is 'fr'."
        },
    )

    def __post_init__(self):
        # Construire le chemin complet vers le fichier audio
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Remonte d'un niveau
        audio_dir = os.path.join(current_dir, "TTSM", "audio")
        self.coqui_speaker_wav = os.path.join(audio_dir, self.coqui_speaker_wav)

        # Vérifier si le fichier existe
        if not os.path.exists(self.coqui_speaker_wav):
            raise FileNotFoundError(f"Le fichier audio {self.coqui_speaker_wav} n'existe pas.")
