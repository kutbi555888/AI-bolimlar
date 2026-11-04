
# scripts/design_model.py
# Vazifasi: YAML konfiguratsiyasini o‘qiydi, GPT arxitekturasini tekshiradi,
# qatlamlar va parametrlar sonini hisoblab, model dizayni haqida hisobot chiqaradi.

from pathlib import Path
import sys
import yaml

# Loyiha asosiy papkasi
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# src modulini import qilish uchun
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_config import ModelConfig


def count_parameters(config):
    """
    GPT arxitekturasining parametrlar sonini hisoblash.

    Hisob quyidagilarni nazarda tutadi:
    - Learned positional embedding
    - LayerNorm
    - QKV attention (bias bilan)
    - GELU Feed Forward Network
    - Bias-siz LM Head
    """

    d = config.n_embd
    v = config.vocab_size
    f = config.ffn_dim

    # 1. Token Embedding
    token_embedding = v * d

    # 2. Position Embedding
    position_embedding = config.block_size * d

    # 3. Attention parametrlari
    qkv = 3 * d * d + 3 * d
    attention_output = d * d + d
    attention = qkv + attention_output

    # 4. Feed Forward Network
    ffn_first = d * f + f
    ffn_second = f * d + d
    ffn = ffn_first + ffn_second

    # 5. Ikkita LayerNorm
    layer_norms = 4 * d

    # 6. Bitta Transformer bloki
    block_params = attention + ffn + layer_norms

    # 7. Barcha Transformer bloklari
    transformer_params = block_params * config.n_layer

    # 8. Final LayerNorm
    final_norm = 2 * d

    # 9. LM Head
    # tie_weights=True bo'lsa, token embedding
    # vaznlari qayta ishlatiladi
    lm_head = 0 if config.tie_weights else v * d

    total_params = (
        token_embedding
        + position_embedding
        + transformer_params
        + final_norm
        + lm_head
    )

    return total_params


def main():

    # YAML konfiguratsiya manzili
    config_path = PROJECT_ROOT / "configs" / "run_01.yaml"

    # YAML faylni o'qish
    with open(config_path, "r", encoding="utf-8") as file:
        yaml_config = yaml.safe_load(file)

    # Model sozlamalarini olish
    config = ModelConfig.from_dict(
        yaml_config["model"]
    )

    # Training sozlamalarini olish
    train_config = yaml_config["training"]

    # Parametrlarni hisoblash
    total_params = count_parameters(config)

    # Model vaznlari uchun taxminiy xotira
    fp32_memory = total_params * 4 / (1024 ** 2)
    fp16_memory = total_params * 2 / (1024 ** 2)

    print("=" * 50)
    print("GPT MODEL DESIGN REPORT")
    print("=" * 50)

    print(f"Vocabulary Size : {config.vocab_size}")
    print(f"Block Size      : {config.block_size}")
    print(f"Layers          : {config.n_layer}")
    print(f"Attention Heads : {config.n_head}")
    print(f"Embedding Dim   : {config.n_embd}")
    print(f"Head Dimension  : {config.head_dim}")
    print(f"FFN Dimension   : {config.ffn_dim}")
    print(f"Dropout         : {config.dropout}")
    print(f"Tie Weights     : {config.tie_weights}")

    print("-" * 50)

    print("MODEL ARCHITECTURE")

    print("1. Token Embedding")
    print("2. Position Embedding")

    for i in range(config.n_layer):
        print(f"3. Transformer Block {i + 1}")
        print("   - LayerNorm")
        print("   - Causal Multi-Head Attention")
        print("   - Residual Connection")
        print("   - LayerNorm")
        print("   - Feed Forward Network")
        print("   - Residual Connection")

    print("4. Final LayerNorm")
    print("5. LM Head")
    print("6. Next Token Logits")

    print("-" * 50)

    print(f"Total Parameters: {total_params:,}")
    print(f"Parameters (M) : {total_params / 1e6:.2f} M")

    print(f"FP32 Weights Memory: {fp32_memory:.2f} MiB")
    print(f"FP16 Weights Memory: {fp16_memory:.2f} MiB")

    print("-" * 50)

    print("TRAINING CONFIGURATION")
    print(f"Epochs       : {train_config['epochs']}")
    print(f"Batch Size   : {train_config['batch_size']}")

    print("=" * 50)
    print("MODEL DESIGN COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()



# Bu kodning asosiy ishlari:
# 1. run_01.yaml faylidan model sozlamalarini o‘qiydi.
# 2. ModelConfig yordamida parametrlar to‘g‘riligini tekshiradi.
# 3. Embedding, Attention, FFN va boshqa qismlarning parametrlar sonini hisoblaydi.
# 4. Transformer arxitekturasining tuzilishini konsolga chiqaradi.
# 5. Model vaznlarini FP32 va FP16 formatlarida saqlash uchun qancha xotira kerakligini taxmin qiladi.
# FP32 — har parametrga 4 bayt ajratiladigan, 32 bitli son formati. FP16 — 2 baytli, 16 bitli format.

# Bu xotira hisobiga gradientlar, optimizer holatlari, aktivatsiyalar va boshqa qo‘shimcha xarajatlar kirmaydi.

# Parametrlar hisobi ham ko‘rsatilgan GPT arxitekturasiga mos. Agar haqiqiy src/model.py boshqacha tuzilsa, parametrlar soni farq qilishi mumkin.