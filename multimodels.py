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
import re
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import joblib
from torch import Tensor

def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing html-tags, punctuation and stopwords

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """    

    text = text.lower()
    text = re.sub('<.*?>', '', text) # html tags
    text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
    text = [word for word in text.split() if word not in stop_words] 
    text = ' '.join(text)
    return text

def get_words_by_freq(sorted_words: list, n: int ) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))

def padding(review_int: list, seq_len: int) -> np.array:
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """    
    features = np.zeros((len(reviews_int), seq_len), dtype = int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)
            
    return features

choice = st.selectbox('Выбери модель', ['Bert', 'LogReg', 'LSTM'])

if choice == 'Bert':
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
        config = torch.load("models/ML+LSTM+Bert/bert/my_tinybert_config.pth")
        
        # Создайте модель
        model = MyTinyBERT(num_classes=config['num_classes'], dropout=config['dropout'])
        
        # Загрузите веса
        model.load_state_dict(torch.load("models/ML+LSTM+Bert/bert/my_tinybert_finetuned.pth", map_location=torch.device('cpu')))
        model.eval()
        
        # Загрузите токенизатор
        tokenizer = AutoTokenizer.from_pretrained("models/ML+LSTM+Bert/bert/tokenuzer")
        
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

elif choice == 'LSTM':
    df = pd.read_json('data/healthcare_facilities_reviews.jsonl',lines=True)
    df = df[['content','sentiment']]
    content = df['content'].tolist()
    preprocessed = [data_preprocessing(content) for content in content]
    corpus = [word for text in preprocessed for word in text.split()]
    sorted_words = Counter(corpus).most_common()
    sorted_words = get_words_by_freq(sorted_words, 200)
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    reviews_int = []
    for text in preprocessed:
        r = [vocab_to_int[word] for word in text.split() if vocab_to_int.get(word)]
        reviews_int.append(r)
    w2v_input = []
    for review in preprocessed:
        cur_review = []
        for word in review.split():
            if vocab_to_int.get(word):
                cur_review.append(word)
        w2v_input.append(cur_review)
    VOCAB_SIZE = len(vocab_to_int) + 1  # размер словаря вместе с токеном padding
    EMBEDDING_DIM = 64 # embedding_dim 
    # Обучим Word2Vec
    wv = Word2Vec(
        vector_size=EMBEDDING_DIM # размерность вектора для слова
        )
    # Сначала word2vec составляет словарь
    wv.build_vocab(w2v_input)
    wv.train(
        corpus_iterable=w2v_input, 
        total_examples=wv.corpus_count, 
        epochs=10
        );
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

    # Бежим по всем словам словаря: если слово есть в word2vec, 
    # достаем его вектор; если слова нет, то распечатываем его и пропускаем
    for word, i in vocab_to_int.items():
        try:
            embedding_vector = wv.wv[word]
            embedding_matrix[i] = embedding_vector
        except KeyError as e:
            pass
            print(f'{e}: word: {word}')
            
    # Создаем предобученный эмбеддинг – этот слой в нашей сети обучаться не будет
    embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
    padded = padding(review_int=reviews_int, seq_len=64)
    X_train, X_valid, y_train, y_valid = train_test_split(
        np.array(padded),
        pd.get_dummies(
            df['sentiment'], 
            drop_first=True
        ).values.astype('int'), test_size=0.2, random_state=1)
    BATCH_SIZE = 64

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE)
    dataiter = iter(train_loader)
    sample_x, sample_y = next(dataiter)
    VOCAB_SIZE = len(vocab_to_int)+1 
    SEQ_LEN = 32
    BATCH_SIZE = 64
    device='cpu'
    HIDDEN_SIZE = 32
    class BahdanauAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.W = nn.Linear(hidden_size, hidden_size)
            self.V = nn.Linear(hidden_size, 1)
        
        def forward(self, hidden, rnn_outputs):
            """
            Args:
                hidden: (batch_size, hidden_size) - последнее скрытое состояние
                rnn_outputs: (batch_size, seq_len, hidden_size) - все выходы RNN
            """
            # hidden: (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
            hidden = hidden.unsqueeze(1)
            
            # Вычисляем скоры внимания
            # rnn_outputs: (batch_size, seq_len, hidden_size)
            # hidden: (batch_size, 1, hidden_size)
            
            scores = torch.tanh(self.W(rnn_outputs) + self.W(hidden))
            scores = self.V(scores).squeeze(-1)  # (batch_size, seq_len)
            
            # Веса внимания
            attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
            
            # Взвешенная сумма - УБЕДИТЕСЬ ЧТО ФОРМЫ ПРАВИЛЬНЫЕ
            # attention_weights: (batch_size, 1, seq_len)
            # rnn_outputs: (batch_size, seq_len, hidden_size)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs)
            context_vector = context_vector.squeeze(1)  # (batch_size, hidden_size)
            
            return context_vector, attention_weights
    from dataclasses import dataclass
    from typing import Union
    @dataclass
    class ConfigRNN:
        vocab_size: int
        device: str
        n_layers: int
        embedding_dim: int
        hidden_size: int
        seq_len: int
        bidirectional: Union[bool, int]
        embedding_dropout: float
    net_config = ConfigRNN(
        vocab_size=len(vocab_to_int) + 1,
        device="cpu",
        n_layers=2,
        embedding_dim=16,
        hidden_size=32,
        seq_len=SEQ_LEN,
        bidirectional=False,
        embedding_dropout=0.2
    )
    class LSTMnetAttention(nn.Module):
        def __init__(self, rnn_conf=net_config):
            super().__init__()
            self.rnn_conf = rnn_conf
            self.vocab_size = rnn_conf.vocab_size
            self.emb_size = rnn_conf.embedding_dim
            self.hidden_dim = rnn_conf.hidden_size
            
            # Увеличиваем размерность эмбеддингов для лучшего захвата семантики
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
            
            # Используем LSTM вместо RNN - лучше запоминает долгосрочные зависимости
            self.rnn = nn.LSTM(
                input_size=self.emb_size,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,  # ВКЛЮЧАЕМ bidirectional для лучшего контекста
                num_layers=2,
                dropout=0.3
            )
            
            # Улучшенный механизм внимания
            self.attention = BahdanauAttention(self.hidden_dim * 2)
            
            # Более глубокая классификационная головка
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(), 
                nn.Dropout(0.3),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            # Embedding
            x = self.embedding(x)
            
            # LSTM с packing для эффективности
            lstm_out, (hidden, cell) = self.rnn(x)
            
            # Для bidirectional LSTM объединяем последние состояния
            if self.rnn.bidirectional:
                last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                last_hidden = hidden[-1]
            
            # Attention с нормализацией
            context_vector, attention_weights = self.attention(last_hidden, lstm_out)
            
            # Классификация
            out = self.classifier(context_vector)
            
            return out, attention_weights  
    model = LSTMnetAttention(net_config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_rnn = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    metric = BinaryAccuracy()


    vocab_to_int = joblib.load("models/ML+LSTM+Bert/lstm/vocab_to_int2.pkl")
    int_to_vocab = {j:i for i, j in vocab_to_int.items()}
    state_dict = torch.load(
            "models/ML+LSTM+Bert/lstm/lstm_sentiment_model.pth", 
            map_location='cpu'
        )
    model.embedding = nn.Embedding(len(vocab_to_int) + 1, 16)

    # Функция для корректной загрузки весов
    def load_custom_embedding(model, state_dict):
        # Создаем новый embedding слой
        new_embedding = nn.Embedding(len(vocab_to_int) + 1, 16)
        
        # Копируем существующие веса
        with torch.no_grad():
            # Копируем веса из старой модели
            new_embedding.weight[:2979] = state_dict['embedding.weight']
            
            # Инициализируем новые токены случайными значениями
            nn.init.xavier_uniform_(new_embedding.weight[2979:])
        
        # Заменяем старый embedding
        model.embedding = new_embedding
        
        # Удаляем embedding.weight из state_dict
        del state_dict['embedding.weight']
        
        # Загружаем остальные параметры модели
        model.load_state_dict(state_dict, strict=False)

    # Применяем функцию
    load_custom_embedding(model, state_dict)
    model.eval()
    # Streamlit интерфейс
    st.title("LSTM Sentiment Analysis - Поликлиники")
    st.write("Введите отзыв для анализа настроения:")

    # Поле ввода
    text = st.text_area("Текст отзыва:", height=150)

    if text:
        # Предобработка
        preprocessed = data_preprocessing(text)
        sequence = [vocab_to_int.get(word, 0) for word in preprocessed.split()]
        
        # Padding
        if len(sequence) < 64:
            sequence.extend([0] * (64 - len(sequence)))
        else:
            sequence = sequence[:64]
        
        # Предсказание
        input_tensor = torch.tensor([sequence], dtype=torch.long)
        with torch.no_grad():
            output, attention_weights = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction = "Positive" if probability > 0.5 else "Negative"
        
        # Результат
        st.subheader("Результат:")
        if prediction == "Positive":
            st.success(f"✅ {prediction} (уверенность: {probability:.4f})")
        else:
            st.error(f"❌ {prediction} (уверенность: {probability:.4f})")
        
        # Детали
        st.write(f"**Обработанный текст:** {preprocessed}")
        st.write(f"**Длина последовательности:** {len(sequence)}")

elif choice == 'LogReg':
    from stop_words import get_stop_words
    def get_improved_russian_stopwords():
        """Улучшенный словарь стоп-слов, сохраняющий негативные контексты"""
        stop_words = set(get_stop_words('russian'))
        
        # УДАЛЯЕМ отрицательные слова из стоп-слов (очень важно!)
        negative_words_to_keep = {
            'не', 'нет', 'ни', 'никак', 'никогда', 'нисколько', 'ничуть',
            'отсутствует', 'плохо', 'ужасно', 'кошмар', 'отвратительно',
            'так себе', 'не очень', 'не понравилось', 'не нравится'
        }
        
        # Убираем отрицательные слова из стоп-слов
        stop_words = stop_words - negative_words_to_keep
        
        # Дополнительные стоп-слова для русского языка
        additional_stopwords = {
            'это', 'вот', 'как', 'так', 'и', 'в', 'над', 'к', 'до',
            'на', 'но', 'за', 'то', 'с', 'ли', 'а', 'во', 'от', 'со',
            'для', 'о', 'же', 'ну', 'вы', 'бы', 'что', 'кто', 'он', 'она'
        }
        
        return stop_words.union(additional_stopwords)
    stop_words = get_improved_russian_stopwords()

    def data_preprocessing(text: str) -> str:
        """preprocessing string: lowercase, removing html-tags, punctuation, 
                                stopwords, digits

        Args:
            text (str): input string for preprocessing

        Returns:
            str: preprocessed string
        """    

        text = text.lower()
        text = re.sub('<.*?>', '', text) # html tags
        text = ''.join([c for c in text if c not in string.punctuation])# Remove punctuation
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([word for word in text.split() if not word.isdigit()]) 
        return text

    def get_words_by_freq(sorted_words: list[tuple[str, int]], n: int = 10) -> list:
        return list(filter(lambda x: x[1] > n, sorted_words))

    def padding(review_int: list, seq_len: int) -> np.array: # type: ignore
        """Make left-sided padding for input list of tokens

        Args:
            review_int (list): input list of tokens
            seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

        Returns:
            np.array: padded sequences
        """    
        features = np.zeros((len(review_int), seq_len), dtype = int)
        for i, review in enumerate(review_int):
            if len(review) <= seq_len:
                zeros = list(np.zeros(seq_len - len(review)))
                new = zeros + review
            else:
                new = review[: seq_len]
            features[i, :] = np.array(new)
                
        return features

    def preprocess_single_string(
        input_string: str, 
        seq_len: int, 
        vocab_to_int: dict,
        verbose : bool = False
        ):
        """Function for all preprocessing steps on a single string

        Args:
            input_string (str): input single string for preprocessing
            seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros
            vocab_to_int (dict, optional): word corpus {'word' : int index}. Defaults to vocab_to_int.

        Returns:
            list: preprocessed string
        """    

        preprocessed_string = data_preprocessing(input_string)
        result_list = []
        for word in preprocessed_string.split():
            try: 
                result_list.append(vocab_to_int[word])
            except KeyError as e:
                if verbose:
                    print(f'{e}: not in dictionary!')
                pass
        result_padded = padding([result_list], seq_len)[0]

        return Tensor(result_padded)
    import pandas as pd

    # Загружаем наш файл с отзывами
    df = pd.read_json('data/healthcare_facilities_reviews.jsonl',lines=True)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder

    vectorizer = CountVectorizer(max_features=5000)  # берем 5000 самых частых слов

    X = vectorizer.fit_transform(df['content'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['sentiment'])

    print(f"Получили матрицу: {X.shape}")
    stop_words = list(stop_words)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words=stop_words)
    X = tfidf.fit_transform(df['content'])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression()
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve)
    from sklearn.model_selection import learning_curve
    import joblib
    vectorizer = joblib.load("models/ML+LSTM+Bert/lm/logreg_model_vectorizer.pkl")
    lr_model = joblib.load("models/ML+LSTM+Bert/lm/logreg_model_model.pkl")
    label_encoder = joblib.load("models/ML+LSTM+Bert/lm/logreg_model_label_encoder.pkl")
    st.title("📊 Анализ отзывов о поликлиниках (LogReg)")
    st.write("Введите отзыв для анализа настроения:")

    # Поле ввода
    text = st.text_area("Текст отзыва:", height=150, 
                    placeholder="Например: 'Отличная поликлиника, врачи внимательные и профессиональные...'")

    if st.button("🎯 Проанализировать") and text:
        if vectorizer is not None and lr_model is not None:
            with st.spinner("Анализируем отзыв..."):
                # 1. Предобработка текста
                processed_text = data_preprocessing(text)
                
                # 2. Преобразуем в TF-IDF
                text_vector = vectorizer.transform([processed_text])
                
                # 3. Предсказание
                prediction = lr_model.predict(text_vector)[0]
                probability = lr_model.predict_proba(text_vector)[0]
                
                # 4. Получаем название класса
                if label_encoder is not None:
                    sentiment = label_encoder.inverse_transform([prediction])[0]
                else:
                    sentiment = "Positive" if prediction == 1 else "Negative"
                
                # 5. Отображаем результат
                st.subheader("📈 Результат анализа:")
                
                if prediction == 1:
                    st.success(f"✅ {sentiment} (уверенность: {probability[prediction]:.2%})")
                else:
                    st.error(f"❌ {sentiment} (уверенность: {probability[prediction]:.2%})")
                
                # 6. Визуализация вероятностей
                st.subheader("📊 Вероятности классов:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    prob_negative = probability[0] if prediction == 0 else 1 - probability[1]
                    st.write("🔴 Negative")
                    st.progress(prob_negative)
                    st.write(f"{prob_negative:.4f}")
                
                with col2:
                    prob_positive = probability[1] if prediction == 1 else 1 - probability[0]
                    st.write("🟢 Positive")
                    st.progress(prob_positive)
                    st.write(f"{prob_positive:.4f}")
                
                # 7. Дополнительная информация
                st.subheader("ℹ️ Детали:")
                st.write(f"**Обработанный текст:** {processed_text}")
                st.write(f"**Размерность вектора:** {text_vector.shape[1]} признаков")
                
        else:
            st.error("❌ Модели не загружены. Проверьте пути к файлам.")
