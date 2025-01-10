import torch
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 1. Cargar los archivos CSV locales. Descargar del dataset de Hugging Face
# https://huggingface.co/datasets/softecapps/spam_ham_spanish/tree/main
dataset = load_dataset("csv", data_files={"train": "../data/train.csv", "test": "../data/test.csv"})

# 2. Cargar el modelo y tokenizer de BETO
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenización del dataset
def tokenize_function(example):
    return tokenizer(example["mensaje"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Preparar los datasets de entrenamiento y evaluación
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# 5. Métricas de evaluación
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 6. Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 7. Inicializar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 8. Ejecutar el Fine-tuning
trainer.train()

# 9. Guardar el modelo entrenado
model.save_pretrained("./beto_fine_tuned")
tokenizer.save_pretrained("./beto_fine_tuned")

# 10. Predicción y generación del classification report
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(-1)

# 11. Mostrar el classification report
report = classification_report(y_true, y_pred, target_names=["ham", "spam"])
print("\nClassification Report:")
print(report)
