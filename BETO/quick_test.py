# Este es un archivo para hacer un quick test con 40 puntos de datos de test

import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report

# 1. Cargar el modelo y el tokenizador desde la carpeta donde los guardaste
model_path = "beto_fine_tuned"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 2. Leer el archivo CSV con los mensajes de prueba
df = pd.read_csv("tests/test_messages.csv")

# 3. Realizar la inferencia sobre los mensajes
predictions = []
for message in df["mensaje"]:
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    predictions.append(prediction)

# 4. Convertir las etiquetas reales y las predicciones a formato numérico
# Etiquetas reales: ham = 0, spam = 1
df["tipo_num"] = df["tipo"].map({"ham": 0, "spam": 1})
y_true = df["tipo_num"].values
y_pred = predictions

# 5. Generar el classification report
report = classification_report(y_true, y_pred, target_names=["ham", "spam"])
print("\nClassification Report:")
print(report)
