from dataclasses import dataclass, field

@dataclass
class MLXLanguageModelHandlerArguments:
    mlx_lm_model_name: str = field(
        default="mlx-community/Phi-3-mini-4k-instruct-8bit",
        metadata={
            "help": "Le modèle de langage pré-entraîné à utiliser."
        },
    )
    mlx_lm_device: str = field(
        default="mps",
        metadata={
            "help": "Le type d'appareil sur lequel le modèle fonctionnera."
        },
    )
    mlx_lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "Le type de données PyTorch pour le modèle et les tenseurs d'entrée."
        },
    )
    mlx_lm_user_role: str = field(
        default="user",
        metadata={
            "help": "Rôle attribué à l'utilisateur dans le contexte de la conversation."
        },
    )
    mlx_lm_init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Rôle initial pour configurer le contexte de la conversation."
        },
    )
    mlx_lm_init_chat_prompt: str = field(
        default="""Tu es Lola une amie virtuelle. Votre tâche est de simuler une conversation naturelle entre deux hommes. Suivez ces directives :
    1. Utilisez un langage familier mais poli, comme dans une conversation entre amis.
    2. Faites des phrases courtes, de 10 à 15 mots maximum.
    3. Réagissez naturellement aux propos de l'utilisateur, comme dans un vrai dialogue.
    4. Posez occasionnellement des questions pour maintenir la conversation.
    5. Utilisez des expressions françaises courantes et du vocabulaire de tous les jours.
    6. Adaptez votre ton et votre style à celui de votre interlocuteur.
    7. Évitez les longues explications ou les détails trop techniques.
    Commencez la conversation de manière décontractée, comme si vous retrouviez un ami.""",
        metadata={
            "help": "Le prompt initial pour établir le contexte de la conversation."
        },
    )
    mlx_lm_gen_max_new_tokens: int = field(
        default=50,
        metadata={
            "help": "Nombre maximum de nouveaux tokens à générer dans une seule réponse."
        },
    )
    mlx_lm_gen_temperature: float = field(
        default=0.7,
        metadata={
            "help": "Contrôle l'aléatoire de la sortie. Une valeur plus élevée rend les réponses plus variées."
        },
    )
    mlx_lm_gen_do_sample: bool = field(
        default=True,
        metadata={
            "help": "Utiliser l'échantillonnage pour plus de variété dans les réponses."
        },
    )
    mlx_lm_chat_size: int = field(
        default=4,
        metadata={
            "help": "Nombre d'interactions assistant-utilisateur à conserver pour le contexte de la conversation."
        },
    )
