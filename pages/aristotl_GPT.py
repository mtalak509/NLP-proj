import streamlit as st
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="GPT Aristotle - Философский генератор текстов",
    page_icon="🏛️",
    layout="wide"
)

# Загрузка модели (кэшируется)
@st.cache_resource
def load_model():
    try:
        model_name = "zhuu4/GPT_aristotle"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Пытаемся загрузить на GPU с оптимизацией памяти
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        except:
            # Если не хватает памяти, грузим на CPU
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
        return tokenizer, model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

# Функция генерации текста
def generate_text(prompt, max_length, num_sequences, temperature, top_k, top_p, repetition_penalty):
    try:
        tokenizer, model = load_model()
        if tokenizer is None or model is None:
            return ["Ошибка загрузки модели"]
        
        # Создаем пайплайн
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Генерация
        results = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return [result['generated_text'] for result in results]
        
    except Exception as e:
        return [f"Ошибка генерации: {str(e)}"]

# Интерфейс
def main():
    # Заголовок
    st.title("🏛️ GPT Aristotle - Философский генератор текстов")
    st.markdown("### Модель, вдохновленная мудростью Аристотеля")
    
    # Сайдбар с настройками
    st.sidebar.header("⚙️ Настройки генерации")
    
    # Примеры промптов
    example_prompts = [
        "О смысле жизни Аристотель говорил:",
        "Добродетель, согласно философии, есть",
        "О политике и государстве:",
        "Природа человека заключается в",
        "Знание и мудность отличаются тем, что"
    ]
    
    # Промпт
    prompt = st.text_area(
        "📝 Введите философский промпт:",
        value="О смысле жизни Аристотель говорил:",
        height=100,
        help="Начальный текст для генерации философского текста"
    )
    
    # Кнопки быстрых промптов
    st.write("🚀 Быстрые промпты:")
    cols = st.columns(len(example_prompts))
    for i, example in enumerate(example_prompts):
        with cols[i]:
            if st.button(example[:20] + "...", key=f"prompt_{i}"):
                st.session_state.prompt = example
                st.rerun()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_length = st.slider(
            "📏 Длина текста:",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Максимальная длина генерируемого текста"
        )
        
        num_sequences = st.slider(
            "🔢 Количество вариантов:",
            min_value=1,
            max_value=5,
            value=1,
            help="Сколько вариантов текста сгенерировать"
        )
    
    with col2:
        temperature = st.slider(
            "🎲 Температура:",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Чем выше, тем более творческий текст"
        )
        
        top_k = st.slider(
            "🎯 Top-k:",
            min_value=0,
            max_value=100,
            value=40,
            help="Ограничивает выбор топ-k токенами (0 = отключено)"
        )
    
    with col3:
        top_p = st.slider(
            "📊 Top-p:",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Nucleus sampling - выбор из топ-p вероятностей"
        )
        
        repetition_penalty = st.slider(
            "🚫 Штраф за повторения:",
            min_value=1.0,
            max_value=2.0,
            value=1.1,
            step=0.1,
            help="Чем выше, тем меньше повторений в тексте"
        )
    
    # Кнопка генерации
    if st.button("🏛️ Сгенерировать философский текст", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("⚠️ Введите промпт для генерации")
            return
            
        with st.spinner("Размышляем над текстом..."):
            results = generate_text(
                prompt, max_length, num_sequences, 
                temperature, top_k, top_p, repetition_penalty
            )
        
        # Отображение результатов
        st.success("✅ Генерация завершена!")
        
        for i, text in enumerate(results, 1):
            with st.expander(f"📜 Философский текст {i}", expanded=True):
                st.text_area(
                    f"Текст {i}",
                    value=text,
                    height=250,
                    key=f"result_{i}",
                    label_visibility="collapsed"
                )
                
                # Кнопка копирования
                st.code(text, language="text")
                
                # Статистика
                chars = len(text)
                words = len(text.split())
                st.caption(f"📊 Символов: {chars}, Слов: {words}")
    
    # Информация в сайдбаре
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ О модели")
    st.sidebar.info("""
    **Модель:** GPT Aristotle  
    **База:** sberbank-ai/rugpt3small_based_on_gpt2  
    **Специализация:** Философские тексты в стиле Аристотеля  
    **Язык:** Русский
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Рекомендации")
    st.sidebar.success("""
    • Температура: 0.7-0.9  
    • Top-p: 0.8-0.9  
    • Длина: 150-300 токенов
    """)
    
    # Статус GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.sidebar.success(f"✅ GPU доступен ({gpu_memory:.1f} GB)")
    else:
        st.sidebar.warning("⚠️ GPU не доступен, используется CPU")

if __name__ == "__main__":
    main()