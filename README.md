# Описание
### Разметка
Разметка данных производилась по ключевым словам. То есть если вопрос вопрос покупателя содержит такие слова как "когда", "вернуть", "сдать", "возврат", "поставк", "акция" и так далее, то
эти записи относяться к классу 0, так как в карточке нет информации об акциях, будущих поставках, опозданиях доставки и других процессах в зоне ответственности wb.
Также если в ответе селлера есть слова "wb", "wildberries", "ситуация", "обратитесь", то такие записи относяться к 0, так как есть отсылка в к процессам не зависящем от продавца.

Напротив, если в ответах селлера содержаться слова "карточка", "размер","комплект", "фото", "да", "нет", то такие записи относяться к классу 1, так как в своем ответе селлер явно ссыллаеться на свою карточку или фотографию в ней
или дает утвердительный ответ на вопрос.
Таким образом получено 3000 строк "очищенных данных". 500(по 250 каждого класса) строк взято как тестовая выборка и не использовалась в обучение.
### Обучение
Были дообучены 3 модели на бинарную классификацию: cointegrated/LaBSE-en-ru, intfloat/multilingual-e5-base и unsloth/Meta-Llama-3.1-8B-bnb-4bit.
Полученные метрики precison и recall на отложенной выборке из 500:

cointegrated/LaBSE-en-ru: (0.918918918918919, 0.952)

intfloat/multilingual-e5-base: (0.9263565891472868, 0.956)

unsloth/Meta-Llama-3.1-8B-bnb-4bit: (0.8764478764478765, 0.908)

В качестве решения взята cointegrated/LaBSE-en-ru, так как она по качеству не сильно отличаеться от e5, но весит сильно меньше.

# Запуск решения
### 1 вариант(локально)
```
git clone <url>

cd Wb_sch_p1

poetry shell

poetry install

python main.py
```

### 2 вариант(в контейнере)

```
git clone <url>

cd Wb_sch_p1

docker build -t <name> .

docker run -p 8000:8000 <name>
```
Теперь по адрессу http://127.0.0.1:8000/docs будет доступен свагер решения.
