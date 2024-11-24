import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, train_loader, eval_loader):
        self.model.train()
        total_loss = 0
        for idx, batch in tqdm(enumerate(train_loader)):
            input_ids = batch["input_ids"].to(self.device, dtype=torch.long)
            attention_mask = batch["input_attention_mask"].to(
                self.device, dtype=torch.long
            )
            labels = batch["label"].to(self.device, dtype=torch.float)

            answer_embedding = batch["answer_embedding"].to(
                self.device, dtype=torch.long
            )
            y_pred = self.model(input_ids, attention_mask, answer_embedding)
            loss = self.loss_fn(y_pred, labels.reshape(-1, 1).float())
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (idx + 1) % 500 == 0:
                print(f"Loss: {total_loss / (idx + 1)}")
                print(f"Evaluating at step {idx + 1}")
                self.evaluate(eval_loader)

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, eval_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(eval_loader)):
                input_ids = batch["input_ids"].to(self.device, dtype=torch.long)
                attention_mask = batch["input_attention_mask"].to(
                    self.device, dtype=torch.long
                )
                labels = batch["label"].to(self.device, dtype=torch.float)

                answer_embedding = batch["answer_embedding"].to(
                    self.device, dtype=torch.long
                )
                y_pred = self.model(input_ids, attention_mask, answer_embedding)
                all_preds.extend(y_pred.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        all_preds = [1 if pred > 0.5 else 0 for pred in all_preds]
        f1 = f1_score(all_labels, all_preds)
        print(f"F1 Score: {f1}")
        self.model.train()
