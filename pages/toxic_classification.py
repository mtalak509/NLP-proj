import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from models.RuBert.func_tools import get_toxic_classification_with_attention, merge_subtokens
import pandas as pd

# ===== НАСТРОЙКА СТРАНИЦЫ =====
st.set_page_config(
    page_title="🔍 Анализатор Токсичности",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CSS СТИЛИ =====
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .toxic-result {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .non-toxic-result {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .attention-bar {
        height: 8px;
        background: linear-gradient(90deg, #4caf50, #ff9800, #f44336);
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== ЗАГОЛОВОК =====
st.markdown('<div class="main-header">🔍 Анализатор Токсичности</div>', unsafe_allow_html=True)
st.markdown("### Определяет токсичные комментарии с помощью AI")

# ===== ЗАГРУЗКА МОДЕЛИ =====
checkpoint = torch.load(
            'models/RuBert/rubert_model_losss_optimized.pth', 
            map_location=torch.device('cpu')
        )

@st.cache_resource
def load_model_and_tokenizer():
    """Загрузка модели и токенизатора с кэшированием"""
    with st.spinner('🔄 Загружаем модель... Это может занять несколько секунд'):
        tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")
        model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-toxicity")
        model.classifier = nn.Linear(312, 2)
        # checkpoint = torch.load(
        #     'models/RuBert/rubert_model_losss_optimized.pth', 
        #     map_location=torch.device('cpu')
        # )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    return model, tokenizer

# ===== ФУНКЦИЯ ВИЗУАЛИЗАЦИИ =====
def create_attention_visualization(result):
    """Создает красивую визуализацию attention весов"""
    words, merged_attention = merge_subtokens(result['tokens'], result['attention_weights'])
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Цвета в зависимости от важности
    mean_attention = np.mean(merged_attention)
    colors = ['#ff6b6b' if w > mean_attention * 1.5 else 
              '#ffa726' if w > mean_attention else 
              '#4ecdc4' for w in merged_attention]
    
    bars = ax.bar(range(len(words)), merged_attention, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Настройки графика
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('🔍 Attention Analysis - Какие слова важны для решения модели', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, merged_attention):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Легенда
    ax.legend(['Высокая важность', 'Средняя важность', 'Низкая важность'], 
              loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    return fig, words, merged_attention

# ===== ОСНОВНОЙ ИНТЕРФЕЙС =====
try:
    # Загрузка модели
    model, tokenizer = load_model_and_tokenizer()
    st.success("✅ Модель успешно загружена!")
    
    # Сайдбар с информацией
    with st.sidebar:
        st.markdown("## ℹ️ О приложении")
        st.info("""
        Это приложение использует модель **RuBERT**, 
        обученную определять токсичные комментарии.
        
        ### Как это работает:
        1. Введите текст в поле ниже
        2. Модель анализирует содержание
        3. Показывает результат с объяснением
        
        ### Метрики:
        - **Accuracy**: 85%+
        - **F1-Score**: 83%+
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Примеры для теста:")
        examples = [
            "Спасибо за помощь, очень полезно!",
            "Ты полный идиот и ничего не понимаешь!",
            "Отличная работа, продолжайте в том же духе",
            "Все кто так думает - дебилы и недоумки"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.text_input = example

    # Основная область
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Введите текст для анализа</div>', unsafe_allow_html=True)
        
        text_input = st.text_area(
            "",
            value=st.session_state.get('text_input', ''),
            placeholder="Введите текст комментария здесь...",
            height=120,
            key="text_input"
        )
        
        analyze_btn = st.button("🚀 Проанализировать текст", type="primary", use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">📊 Статистика модели</div>', unsafe_allow_html=True)
        
        

        chart_data = pd.DataFrame({
        'Train Loss': checkpoint['train loss'],
        'Validation Loss': checkpoint['valid loss']})
    
        # Красивый line chart
        st.line_chart(
        chart_data,
        color=['#FF6B6B', '#4ECDC4'],  # Красный для train, бирюзовый для validation
        height=400)

        st.markdown("""
        <div class="metric-card">
            <h3>85.3%</h3>
            <p>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h3>85.1%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    # Анализ текста
    if analyze_btn and text_input:
        with st.spinner('🔍 Анализируем текст...'):
            result = get_toxic_classification_with_attention(text_input, model, tokenizer)
            
            # Создаем визуализацию
            fig, words, attention_weights = create_attention_visualization(result)
            
            # Отображаем результаты
            st.markdown("---")
            st.markdown('<div class="sub-header">📊 Результаты анализа</div>', unsafe_allow_html=True)
            
            # Основной результат
            col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
            
            with col_res1:
                if result['prediction'] == 1:
                    st.markdown("""
                    <div class="result-box toxic-result">
                        <h3>🚨 ТОКСИЧНЫЙ</h3>
                        <p>Текст содержит оскорбительные выражения</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box non-toxic-result">
                        <h3>✅ НЕТОКСИЧНЫЙ</h3>
                        <p>Текст вежливый и уважительный</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_res2:
                confidence = result['probability']
                st.metric(
                    label="Уверенность модели",
                    value=f"{confidence:.1%}",
                    delta="высокая" if confidence > 0.8 else "средняя" if confidence > 0.6 else "низкая"
                )
                
                # Прогресс-бар токсичности
                st.markdown("**Уровень токсичности:**")
                st.progress(float(confidence if result['prediction'] == 1 else 1 - confidence))
            
            with col_res3:
                st.metric(
                    label="Вероятность токсичности",
                    value=f"{result['probability']:.1%}" if result['prediction'] == 1 else f"{(1 - result['probability']):.1%}",
                    delta="опасно" if result['prediction'] == 1 and result['probability'] > 0.7 else "нормально"
                )

            # Визуализация attention
            st.markdown("---")
            st.markdown('<div class="sub-header">🔍 Анализ внимания модели</div>', unsafe_allow_html=True)
            st.pyplot(fig)
            
            # Детальная информация о словах
            with st.expander("📋 Детальная информация о словах"):
                mean_attention = np.mean(attention_weights)
                st.write("**Самые важные слова для решения модели:**")
                
                # Сортируем слова по важности
                sorted_indices = np.argsort(attention_weights)[::-1]
                
                for i, idx in enumerate(sorted_indices[:5]):  # Топ-5 слов
                    word = words[idx]
                    weight = attention_weights[idx]
                    importance = "🔴 Высокая" if weight > mean_attention * 1.5 else "🟡 Средняя" if weight > mean_attention else "🟢 Низкая"
                    
                    col_word, col_weight, col_imp = st.columns([2, 1, 1])
                    with col_word:
                        st.write(f"**{word}**")
                    with col_weight:
                        st.write(f"{weight:.4f}")
                    with col_imp:
                        st.write(importance)

    elif analyze_btn and not text_input:
        st.warning("⚠️ Пожалуйста, введите текст для анализа")

except Exception as e:
    st.error(f"❌ Произошла ошибка: {str(e)}")
    st.info("🔧 Проверьте наличие файлов модели и правильность структуры проекта")

# ===== ФУТЕР =====
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔍 Анализатор токсичности | Powered by RuBERT | "
    "<a href='https://github.com/your-repo' target='_blank'>GitHub</a>"
    "</div>", 
    unsafe_allow_html=True
)