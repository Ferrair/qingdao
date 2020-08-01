from src.model.tail import TailModel

last_temp_1 = 135.55
last_temp_2 = 120.323

tail_model = TailModel(next_range_1=132.21, next_range_2=117.3)
for _ in range(600):
    pred = tail_model.predict("Batch", 1999, last_temp_1, last_temp_2)
    last_temp_1 = pred[0]
    last_temp_2 = pred[1]
    print(last_temp_1, last_temp_2)
