
# src/model_config.py
# Vazifasi: Modelning tuzilishini belgilaydi, konfiguratsiya parametrlarini saqlaydi
#  va ularning to‘g‘riligini tekshiradi.


from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Kichik GPT modelining arxitektura sozlamalari.
    """

    # Tokenizer lug'atidagi tokenlar soni
    vocab_size: int = 8000

    # Bitta ketma-ketlikning maksimal uzunligi
    block_size: int = 1024

    # Transformer bloklari soni
    n_layer: int = 6

    # Attention Headlar soni
    n_head: int = 8

    # Token embedding o'lchami
    n_embd: int = 256

    # Feed Forward Network kengayish koeffitsiyenti
    ffn_mult: int = 4

    # Overfittingni kamaytirishga yordam beruvchi dropout
    dropout: float = 0.1

    # Token Embedding va LM Head vaznlarini bog'lash
    tie_weights: bool = True

    def __post_init__(self):
        """
        Konfiguratsiya to'g'riligini tekshiradi.
        """

        if self.vocab_size <= 0:
            raise ValueError("vocab_size musbat bo'lishi kerak")

        if self.block_size <= 0:
            raise ValueError("block_size musbat bo'lishi kerak")

        if self.n_layer <= 0:
            raise ValueError("n_layer musbat bo'lishi kerak")

        if self.n_head <= 0:
            raise ValueError("n_head musbat bo'lishi kerak")

        if self.n_embd <= 0:
            raise ValueError("n_embd musbat bo'lishi kerak")

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd n_head ga qoldiqsiz bo'linishi kerak"
            )

        if self.ffn_mult <= 0:
            raise ValueError("ffn_mult musbat bo'lishi kerak")

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout 0 va 1 oralig'ida bo'lishi kerak"
            )

    @property
    def head_dim(self):
        """Har bir Attention Head o'lchami."""
        return self.n_embd // self.n_head

    @property
    def ffn_dim(self):
        """Feed Forward Network ichki o'lchami."""
        return self.n_embd * self.ffn_mult

    @classmethod
    def from_dict(cls, config_dict):
        """Dictionary ma'lumotidan ModelConfig yaratadi."""
        return cls(**config_dict)


# Asosiy parametrlar:
# Parametr	Qiymat	Vazifasi
# vocab_size	8000	Tokenlar lug‘ati hajmi
# block_size	1024	Maksimal kontekst uzunligi (tokenlarda)
# n_layer	6	Transformer bloklari soni
# n_head	8	Attention Headlar soni
# n_embd	256	Tokenlarning vektor o‘lchami
# ffn_mult	4	FFNning kengayish koeffitsiyenti
# dropout	0.1	Trainingda tasodifiy faolliklarni o‘chirish ehtimoli
# tie_weights	True	Embedding va LM Head vaznlarini birgalikda ishlatish

# Bu fayl modelning konfiguratsiyasini belgilaydi, lekin hali Transformer qatlamlarining o‘zini yaratmaydi.
#  Haqiqiy model arxitekturasi keyinchalik src/model.py ichida yoziladi.



