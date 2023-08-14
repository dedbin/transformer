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

def encode(text: str, tokenizer:any) -> torch.Tensor:
    """Функция для кодирования входного текста с использованием предварительно обученного токенизатора и векторизованных поисковых запросов"""
    ...

def decode(enc_sec: torch.Tensor,  tokenizer:any) -> str
    """Функция для декодирования входной последовательности в текст"""
    ...

def get_batch(date: list[str], block_size: int, batch_size: int) -> tuple[any,any]:
    """Это простая функция для создания батча.
    GPUs обрабатывать паралельно, поэтому можно загружать несколько блоков одновременно,
     поэтому нужны батчи - сколько независимых последовательностей будут обработаны параллельно."""
    ...

def load_model_from_checkpoint(model: torch.nn.Module, checkpoint_path: str = "checkpoints/state_dict_model.pt", **kwargs) -> torch.nn.Module:
    """Функция для загрузки модели из файла"""
    ...

def save_model_to_checkpoint(model: torch.nn.Module, checkpoint_path: str = "checkpoints/state_dict_model.pt", epoch: int = 0) -> None:
    """Функция для сохранения модели в файл"""
    ...