"""
Fine-tune BERTimbau (neuralmind/bert-base-portuguese-cased) para
classificar respostas de analistas como completas ou incompletas.
Usa o mesmo CSV sintético que geramos anteriormente.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

CSV_PATH = "base_manifestacoes.csv"          # <<-- troque se necessário
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NUM_EPOCHS = 3
BATCH_SIZE = 4
OUTPUT_DIR = "./bert_resposta_completa"

# 1. Carregar base (ou sua base real)
df = pd.read_csv(CSV_PATH)
df["texto"] = df["manifestacao"] + " " + df["resposta"]
label2id = {"resposta_completa": 1, "resposta_incompleta": 0}
df["label"] = df["classificacao"].map(label2id)

# 2. Split
train_df, test_df = train_test_split(df, test_size=0.25,
                                     stratify=df["label"], random_state=42)

# 3. Tokenização
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tok(batch["texto"], truncation=True, padding="max_length",
               max_length=256)

from datasets import Dataset
train_ds = Dataset.from_pandas(train_df[["texto", "label"]]).map(tokenize,
                    batched=True, remove_columns=["texto"])
test_ds  = Dataset.from_pandas(test_df [["texto", "label"]]).map(tokenize,
                    batched=True, remove_columns=["texto"])

# 4. Modelo
model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, id2label={v:k for k,v in label2id.items()},
            label2id=label2id)

# 5. Treinamento
args = TrainingArguments(
        OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss")

trainer = Trainer(model, args, train_ds, test_dataset=test_ds)

trainer.train()

# 6. Avaliação
preds = trainer.predict(test_ds)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(1)

print(classification_report(y_true, y_pred,
      target_names=["resposta_incompleta","resposta_completa"]))

print("Matriz de confusão:\n", confusion_matrix(y_true, y_pred))

# 7. Salvamento do modelo
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)

# Usar depois do treino
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "./bert_resposta_completa"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def classificar(manifestacao, resposta):
    texto = manifestacao + " " + resposta
    inputs = tok(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(1).item()
    return "resposta_completa" if pred==1 else "resposta_incompleta"
