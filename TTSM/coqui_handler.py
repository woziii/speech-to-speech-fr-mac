from TTS.api import TTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch
import os
import tempfile
import re

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

console = Console()

class CoquiTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        language="fr",
        speaker_wav="female.wav",
        device="cpu",
        gen_kwargs={},
        blocksize=512,
        max_sentence_length=150,  # Nouveau paramètre pour la longueur maximale des phrases
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language
        self.speaker_wav = speaker_wav
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs
        self.max_sentence_length = max_sentence_length  # Nouvelle variable d'instance

        logger.info(f"Initializing Coqui TTS with model: {model_name}")
        self.model = TTS(model_name=model_name).to(device)

        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Temporary directory for audio files: {self.temp_dir}")

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        temp_file = os.path.join(self.temp_dir, "female.wav")
        self.model.tts_to_file(text="Warmup text", speaker_wav=self.speaker_wav, language=self.language, file_path=temp_file)
        os.remove(temp_file)

    def split_sentence(self, sentence):
        # Fonction pour découper une phrase en morceaux plus petits
        words = sentence.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(" ".join(current_chunk + [word])) > self.max_sentence_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
            current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process(self, llm_sentence):
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        if self.device == "mps":
            import time
            start = time.time()
            torch.mps.synchronize()
            torch.mps.empty_cache()
            _ = time.time() - start

        # Découper la phrase en morceaux plus petits
        sentence_chunks = self.split_sentence(llm_sentence)

        for chunk in sentence_chunks:
            temp_file = os.path.join(self.temp_dir, f"output_{hash(chunk)}.wav")
            self.model.tts_to_file(text=chunk, speaker_wav=self.speaker_wav, language=self.language, file_path=temp_file)

            audio_chunk, _ = librosa.load(temp_file, sr=16000)
            os.remove(temp_file)

            if len(audio_chunk) == 0:
                continue

            audio_chunk = (audio_chunk * 32768).astype(np.int16)

            for i in range(0, len(audio_chunk), self.blocksize):
                yield np.pad(
                    audio_chunk[i : i + self.blocksize],
                    (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
                )

        self.should_listen.set()

    def __del__(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
