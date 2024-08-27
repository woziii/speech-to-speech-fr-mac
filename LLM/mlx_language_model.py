import logging
from LLM.chat import Chat
from baseHandler import BaseHandler
from mlx_lm import load, stream_generate, generate
from rich.console import Console
import torch
import signal

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

console = Console()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout_handler)

class MLXLanguageModelHandler(BaseHandler):
    def setup(
        self,
        model_name="mlx-community/Phi-3-mini-4k-instruct-4bit",
        device="mps",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        logger.info(f"Loading MLX model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        self.gen_kwargs = gen_kwargs

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError("An initial prompt needs to be specified when setting init_chat_role.")
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role

        self.warmup()
        logger.info("MLX model setup complete")

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_input_text = "Write me a poem about Machine Learning."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        n_steps = 2
        for _ in range(n_steps):
            prompt = self.tokenizer.apply_chat_template(dummy_chat, tokenize=False)
            generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.gen_kwargs["max_new_tokens"],
                verbose=False,
            )
        logger.info("Warmup complete")

    def process(self, prompt):
        logger.info(f"MLX LM processing input: {prompt}")
        self.chat.append({"role": self.user_role, "content": prompt})

        chat_messages = [msg for msg in self.chat.to_list() if msg["role"] != "system"]
        prompt = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        
        try:
            signal.alarm(30)  # 30 seconds timeout
            output = ""
            curr_output = ""
            for t in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.gen_kwargs["max_new_tokens"],
            ):
                logger.debug(f"Generated token: {t}")
                output += t
                curr_output += t
                if curr_output.endswith((".", "?", "!", "<|end|>")):
                    yield curr_output.replace("<|end|>", "")
                    curr_output = ""
            signal.alarm(0)
        except TimeoutException:
            logger.error("LM processing timed out")
            yield "Je suis désolé, mais je n'ai pas pu générer une réponse à temps."
            return

        generated_text = output.replace("<|end|>", "")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self.chat.append({"role": "assistant", "content": generated_text})
        logger.info("MLX LM finished processing")