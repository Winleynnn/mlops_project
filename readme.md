# mlops_project
### Таблица обновлений проекта (будет обновляться по ходу выполнения)
|Дата|Название|Краткое описание|
|----|--------|--------|
|23.02.2024|Составлен readme.md|Добавлено текстовое описание проекта: используются временные ряды [погодных данных](https://open-meteo.com/en/docs/historical-weather-api) для прогнозирования метеорологических условий, с помощью [LSTM-nn](https://www.tensorflow.org/tutorials/structured_data/time_series#recurrent_neural_network).
## Описание проекта:
  1. Формулировка задачи: прогнозирование погодных условий с помощью временных рядов метеоданных. Полученные прогнозы можно использовать для различных целей:
        - Определение потенциальных погодных условий для корректировки стиля верхней одежды
        - Корректировка мелиоративных процессов и управленческих решений (в контексте сельского хозяйства)
        - и другие.
  2. Для формирования качественного прогноза погоды необходимы данные за достаточно длительный период, чтобы учесть как сезонность динамики данных, так и изменения в самой сезонности (например, учета климатических изменений). Предполагается использование обновляющихся исторических погодных данных, представленных в [Open Meteo](https://open-meteo.com/en/docs/historical-weather-api). Доступ к данным формируется с помощью запроса к API, содержатся записи с 1940 года по текущий момент времени, общим объемом около 90 терабайт. Поскольку архив обновляется с 5-дневной задержкой, акуальные данные будут подягиваться через этот [API](https://open-meteo.com/en/docs). Данные обладают следующими особенностями/проблемами:
        - Пространственная точность варьируется от 25 до 9 км (в зависимости от временного периода), что в некоторых случаях может приводить к неточному прогнозу (например, когда данные для одного района берутся из соседнего, в котором, например, не выпадали осадки, что сильно влияет на уровень влажности)
        - Пропуски в данных уже заполнены, что, с одной стороны, облегчает работу с данными, с другой - может серьезно повлиять на валидность прогноза на основе таких данных, поскольку неизвестно, как именно эти пропуски заполнялись. Также неизвестна информация касательно того, какое количество записей отсутствало для конкретной точки в пространстве
  3. Концепцию пайплайна обучения (для обработки, анализа, работы с временными рядами, обучения нейросетей) предполагается использовать из [этого туториала по прогнозированию временных рядов](https://www.tensorflow.org/tutorials/structured_data/time_series#setup), с той лишь разницей, что вместо tensorflow будет применяться pytorch, а период прогнозирования может быть увеличен, поскольку используемый датасет содержит записи за значительно больший период. Наилучшие метрики по качеству прогнозирования в туториале у LSTM-нейросети, поэтому предполагается использование идентичной архитектуры(разумеется, с учетом разницы в датасетах): ![](https://www.tensorflow.org/static/tutorials/structured_data/images/lstm_many_window.png)
  4. После обучения модели необходимо периодически ее дообучать, поскольку временной ряд будет постоянно пополняться - предлагается дообучать ее каждые сутки (раз в 24 часа). Схема взаимодействия пользователя с сервисом выглядит следующим образом:
  ![](scheme.png)
Поскольку API имеет доступ к большому количеству локаций, стоит ограничить пользовательский выбор возможных локаций, осуществляться который будет с помощью поиска по интерактивной карте, либо же с помощью поиска по названию. Для отправки запросов к обученной нейросети и формированию прогнозов будет использован Flask в связи с API OpenStreetMap.