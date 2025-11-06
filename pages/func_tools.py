import matplotlib.pyplot as plt
import numpy as np
import torch

def get_toxic_classification_with_attention(text: str, model, tokenizer, device='cpu'):
    """
    Классификация с возвратом attention весов
    
    Returns:
        dict: предсказание и attention веса
    """
    # Токенизация
    tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        # Для получения attention весов нужно использовать output_attentions=True
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=True  # важно!
        )
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Получаем предсказание
        prediction = torch.argmax(logits, dim=1).item()
        probability = torch.softmax(logits, dim=1)[0][prediction].item()
        
        # Извлекаем attention веса
        attentions = outputs.attentions  # tuple с attention всех слоев
        
        # Берем attention из последнего слоя
        last_layer_attention = attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
        
        # Усредняем по всем головам внимания
        avg_attention = last_layer_attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Берем attention для [CLS] токена (как он "смотрит" на другие токены)
        cls_attention = avg_attention[0, 0, :]  # [seq_len]
        
        # Получаем токены
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return {
            'prediction': prediction,
            'probability': probability,
            'class_name': 'toxic' if prediction == 1 else 'non-toxic',
            'tokens': tokens,
            'attention_weights': cls_attention.cpu().numpy()
        }
    


def merge_subtokens(tokens, attention_weights):
    """
    Объединяет субтокены обратно в слова и суммирует их attention веса
    
    Args:
        tokens: список токенов
        attention_weights: список attention весов
    
    Returns:
        tuple: (слова, объединенные attention веса)
    """
    words = []
    merged_attention = []
    current_word = ""
    current_attention = 0.0
    count = 0
    
    for token, attention in zip(tokens, attention_weights):
        # Пропускаем служебные токены
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_word:
                words.append(current_word)
                merged_attention.append(current_attention / count if count > 0 else current_attention)
                current_word = ""
                current_attention = 0.0
                count = 0
            continue
        
        # Если токен начинается с ## - это продолжение слова
        if token.startswith('##'):
            current_word += token[2:]  # убираем ##
            current_attention += attention
            count += 1
        else:
            # Если есть текущее слово, сохраняем его
            if current_word:
                words.append(current_word)
                merged_attention.append(current_attention / count if count > 0 else current_attention)
            
            # Начинаем новое слово
            current_word = token
            current_attention = attention
            count = 1
    
    # Добавляем последнее слово
    if current_word:
        words.append(current_word)
        merged_attention.append(current_attention / count if count > 0 else current_attention)
    
    return words, merged_attention




def visualize_attention_merged(text: str, model, tokenizer, device='cpu'):
    """
    Визуализация attention весов с объединенными словами
    """
    result = get_toxic_classification_with_attention(text, model, tokenizer, device)
    
    tokens = result['tokens']
    attention_weights = result['attention_weights']
    
    # Объединяем субтокены в слова
    words, merged_attention = merge_subtokens(tokens, attention_weights)
    
    # Создаем график
    fig = plt.figure(figsize=(12, 6))
    
    # Определяем цвета для выделения важных слов
    mean_attention = np.mean(merged_attention)
    colors = ['red' if w > mean_attention else 'blue' for w in merged_attention]
    
    bars = plt.bar(range(len(words)), merged_attention, color=colors, alpha=0.7)
    plt.xticks(range(len(words)), words, rotation=45, ha='right', fontsize=10)
    plt.title(f'Attention Weights - Prediction: {result["class_name"].upper()} (confidence: {result["probability"]:.3f})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, merged_attention):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # # Выводим слова с attention весами
    # print("Слова с attention весами:")
    # print("-" * 40)
    # for word, weight in zip(words, merged_attention):
    #     importance = "🔴 ВАЖНО" if weight > mean_attention else "🔵 нормально"
    #     print(f"{word:15} {weight:.4f} {importance}")
    
    # print(f"\nСредний attention: {mean_attention:.4f}")
    # print(f"Предсказание: {result['class_name']} (вероятность: {result['probability']:.3f})")
    
    # # Добавляем объединенные данные в результат
    # result['merged_words'] = words
    # result['merged_attention'] = merged_attention
    
    return result, fig