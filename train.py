import torch
from model import Transformer
from utils import *
from transformers import AutoTokenizer

# загрузка модели из файла
# m = load_model_from_checkpoint(Transformer,vocab_size=vocab_size)

path_do_data = "data/text_to_train.txt"
data_raw = open(path_do_data, encoding="utf-8").read()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

data = encode(text=data_raw, tokenizer=tokenizer)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = Transformer(
    vocab_size=vocab_size,
    num_embed=NUM_EMBED,
    block_size=BLOCK_SIZE,
    num_heads=NUM_HEAD,
    num_layers=NUM_LAYER,
    dropout=DROPOUT,
)

m = model.to(DEVICE)

print(f"Model with {(sum(p.numel() for p in m.parameters()) / 1e6):.2f}M parameters")

optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITER):
    if step % EVAL_INTERVAL == 0 or step == MAX_ITER - 1:
        loss_train = estimate_loss(
            data=train_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        loss_val = estimate_loss(
            data=val_data, model=m, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE
        )
        print(f"step {step:.2f} | train loss {loss_train:6.4f} | val loss {loss_val:6.4f}")

        xb, yb = get_batch(data=train_data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
        logits, loss = m.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_model_to_checkpoint(model=m, checkpoint_path="checkpoints", epoch=step)

    # generate some output based on the context
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(
        decode(
            enc_sec=m.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
            tokenizer=tokenizer,
        )
    )