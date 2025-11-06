import gdown
import pandas as pd
import os

def download_csv_from_drive(url, output_path='data.csv'):
    """
    Скачивает CSV файл с Google Drive
    
    Args:
        url (str): Ссылка на Google Drive
        output_path (str): Путь для сохранения файла
    """
    # Преобразуем URL в формат для скачивания
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Скачиваем файл
    gdown.download(download_url, output_path, quiet=False)
    
    # Загружаем в DataFrame для проверки
    df = pd.read_csv(output_path)
    print(f"Файл успешно скачан и сохранен как: {output_path}")
    print(f"Размер данных: {df.shape}")
    
    return df

# Использование
url = "https://drive.google.com/file/d/1O7orH9CrNEhnbnA5KjXji8sgrn6iD5n-/view"
df = download_csv_from_drive(url, 'my_dataset.csv')