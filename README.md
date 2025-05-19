# Прунинг на основе градиентов дял удаления концептов

## Установка
```bash
git clone https://github.com/Dgeka24/gradients_pruning_erasing.git
cd gradients_pruning_erasing
pip install -r requirements.txt
```

## Запуск метода
```bash
python prune.py --concept_to_erase 'ship'
```

## Генерация изображений
```bash
python generate_images.py --prompt 'a photo of ship'
```