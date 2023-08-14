import os
import torch
from datetime import datetime

BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_ITER = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_HEAD = 6
NUM_EMBED = NUM_HEAD * 128
NUM_LAYER = 6
DROPOUT = 0.2


def encode(text: str, tokenizer: any) -> torch.Tensor:
    """Функция для кодирования входного текста с использованием предварительно обученного токенизатора и векторизованных поисковых запросов"""
    tokens = tokenizer.tokenize(text)
    token_indices = tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor(token_indices, dtype=torch.long)


def decode(enc_sec: torch.Tensor,  tokenizer:any) -> str:
    """Функция для декодирования входной последовательности в текст"""
    enc_sec = enc_sec.tolist()
    return tokenizer.decode(enc_sec)


def get_batch(data: list[str], block_size: int, batch_size: int) -> tuple[any, any]:
    """Это простая функция для создания батча.
    GPUs обрабатывать паралельно, поэтому можно загружать несколько блоков одновременно,
     поэтому нужны батчи - сколько независимых последовательностей будут обработаны параллельно."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


def load_model_from_checkpoint(
        model_cls: torch.nn.Module,
        checkpoint_path: str = "checkpoints/state_dict_model.pt",
        **kwargs: dict) -> torch.nn.Module:
    """Функция для загрузки модели из файла"""
    try:
        state_dict = torch.load(checkpoint_path)
        print("Загрузка модели из файла завершена")
        model = model_cls(**kwargs)
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели из файла {e}")


def save_model_to_checkpoint(model: torch.nn.Module, checkpoint_path: str = "checkpoints/state_dict_model.pt",
                             epoch: int = 0) -> None:
    """Функция для сохранения модели в файл"""
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    now = datetime.now()
    current_date = now.strftime("%d.%m.%Y_%H:%M:%S")
    checkpoint_name = f"checkpoint_epoch-{epoch}_{current_date}.pt"
    full_path = os.path.join(checkpoint_path, checkpoint_name)
    try:
        torch.save(model.state_dict(), full_path)
        print(f"Модель сохранена в файл {full_path}")
    except Exception as e:
        print(f"Ошибка при сохранении модели в файл {e}")
@torch.no_grad()
def estimate_loss(
        data: list[str],
        model: torch.nn.Module,
        block_size: int,
        batch_size: int,
        iters: int = 10
):
    """Функция для оценки потерь"""
    out = {}
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(data, block_size, batch_size)
        logits, loss = model.forward(x, y)
        losses[i] = loss.item()
    out = losses.mean()
    model.train()
    return out
