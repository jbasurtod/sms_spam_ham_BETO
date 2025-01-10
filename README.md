
# Fine-Tuning de BETO para Detección de Fraude en Mensajes de Texto (Spam vs Ham)

## 📚 Descripción del Proyecto
Este proyecto realiza el **fine-tuning del modelo BETO (BERT en Español)** para la tarea de **detección de mensajes fraudulentos (spam)** en español. Utilizando un dataset de **Hugging Face** y ampliándolo en un **40%** con datos sintéticos, se busca mejorar la capacidad del modelo para distinguir entre mensajes **legítimos (ham)** y **fraudulentos (spam)**.

El modelo final detecta con **un recall del 81% los mensajes spam**, priorizando la identificación de fraudes en diferentes tipos de mensajes de texto.

---

## 📦 Dataset Utilizado
El dataset base proviene de Hugging Face y está disponible en el siguiente enlace:

🔗 [**Dataset: spam_ham_spanish**](https://huggingface.co/datasets/softecapps/spam_ham_spanish/tree/main)

Este dataset contiene mensajes etiquetados como **ham** o **spam** en español. Se amplió en un **40%** con mensajes generados sintéticamente para mejorar el rendimiento del modelo.

---

## ⚙️ Arquitectura del Modelo
El modelo utilizado es **BETO**:

- **Modelo base:** `dccuchile/bert-base-spanish-wwm-uncased`
- **Tokenizador:** BertTokenizer
- **Framework:** Hugging Face Transformers

El fine-tuning se realizó utilizando **PyTorch** y el **Hugging Face Trainer API**.

---

## 🧪 Resultados Clave
| Métrica       | ham      | spam     | Total  |
|---------------|----------|----------|--------|
| **Precision** | 0.71     | 0.65     | 0.68   |
| **Recall**    | 0.53     | 0.81     | 0.68   |
| **F1-Score**  | 0.61     | 0.72     | 0.67   |

- **Recall para la clase spam:** 81%  
   Esto significa que el modelo detecta **la mayoría de los mensajes fraudulentos**, lo cual es esencial en tareas de detección de fraude.

- **Precisión general:** 68%  
   Se buscará mejorar esta métrica en iteraciones futuras.

---

## 🚀 Cómo Ejecutar el Proyecto

1. Clona este repositorio:

```bash
git clone git@github.com:jbasurtod/nlp_fraud_detection.git
```

2. Instala las dependencias necesarias:

```bash
pip install torch transformers datasets scikit-learn
```

3. Descarga los datasets:

- `train.csv`
- `test.csv`

4. Ejecuta el entrenamiento del modelo:

```python
from transformers import Trainer

# Código de entrenamiento
trainer.train()
```

---

## 📈 Próximos Pasos
- Aumentar la precisión general del modelo.
- Probar con técnicas de **data augmentation** para mensajes legítimos (ham).
- Ajustar los hiperparámetros para mejorar el balance entre precisión y recall.

---

## 🖋️ Autor
Proyecto realizado por **Juan Carlos Basurto**.  
Si tienes alguna pregunta o sugerencia, ¡no dudes en contactarme!
