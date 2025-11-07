import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torchutils as tu
from sklearn.model_selection import train_test_split
from collections import Counter
from torchmetrics.classification import BinaryAccuracy
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import time


class MyTinyBERT(torch.nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        hidden_size = self.bert.config.hidden_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.bert.device)
        attention_mask = batch['attention_mask'].to(self.bert.device)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    # Загрузите конфигурацию
    config = torch.load("/home/nikita/ds-phase-2/10-nlp/NLP-proj/models/ML+LSTM+Bert/my_tinybert_config.pth")
    
    # Создайте модель
    model = MyTinyBERT(num_classes=config['num_classes'], dropout=config['dropout'])
    
    # Загрузите веса
    model.load_state_dict(torch.load("/home/nikita/ds-phase-2/10-nlp/NLP-proj/models/ML+LSTM+Bert/my_tinybert_finetuned.pth", map_location=torch.device('cpu')))
    model.eval()
    
    # Загрузите токенизатор
    tokenizer = AutoTokenizer.from_pretrained("/home/nikita/ds-phase-2/10-nlp/NLP-proj/models/ML+LSTM+Bert/tokenuzer")
    
    return model, tokenizer

# Заголовок приложения
st.title("BERT Text Classifier")
st.write("Введите текст для классификации:")

# Загрузите модель и токенизатор при старте
model, tokenizer = load_model()


# Поле ввода текста
text = st.text_area("Текст", height=150)


if text:
    start_time = time.time()  # Засекаем время начала

    # Токенизация и инференс (ваш существующий код)
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    end_time = time.time()
    inference_time = end_time - start_time

    # Определяем соответствие классов меткам
    class_labels = {0: "негативный", 1: "положительный"}
    predicted_label = class_labels[predicted_class]  # преобразуем номер класса в метку

    # Вывод результата с новыми метками
    st.write(f"**Предсказанный класс:** {predicted_label}")
    st.write(f"**Уверенность:** {confidence:.4f}")

    # Дополнительно: вероятности по классам (с подписями)
    st.write("**Вероятности по классам:**")

    # Создаём колонки для параллельного отображения текста и прогресс-баров
    cols = st.columns(2)

    for i, prob in enumerate(probabilities[0]):
        prob_rounded = round(prob.item(), 4)
        label = class_labels[i]  # используем метку вместо номера класса

        # Колонка 1: текст с вероятностью
        with cols[0]:
            st.write(f"{label}")

        # Колонка 2: прогресс-бар
        with cols[1]:
            st.progress(prob_rounded)  # прогресс-бар от 0 до 1
            st.caption(f"{prob_rounded:.4f}")  # числовое значение под баром

    # Таблица с вероятностями (дополнительная визуализация) с текстовыми метками
    st.dataframe(
        pd.DataFrame({
            'Класс': [class_labels[i] for i in range(len(probabilities[0]))],  # заменяем номера классов на метки
            'Вероятность': [round(p.item(), 4) for p in probabilities[0]]
        }),
        hide_index=True,
        use_container_width=True
    )

    # Выделение доминирующего класса с текстовой меткой
    max_probs, max_idxs = torch.max(probabilities, dim=1)  # dim=1 — по строкам (классам)
    max_prob = torch.max(probabilities[0])
    max_idx = torch.argmax(probabilities[0]).item()
    final_label = class_labels[max_idx]  # преобразуем номер класса в метку

    st.success(f"**Финальный прогноз:** {final_label} (уверенность: {max_prob.item():.4f})     Время: {inference_time:.4f}")