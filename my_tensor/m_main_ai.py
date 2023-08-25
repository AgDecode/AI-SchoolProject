from keras.layers import Dense
from keras.models import Sequential
import numpy as np


def reverse_encode(np_list):
    # example_input - [1.4149497e-01 8.8076782e-01 2.3201625e-01 4.9572612e-01 5.8198853e-01
    #  1.3676369e-01]

    max_number_index = np_list.index(max(np_list))
    np_list[max_number_index] = 1.0
    for i in range(len(np_list)):
        if np_list[i] != 1.0:
            np_list[i] = 0.0
    # example_output - [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    answer_Index = np_list.index(1.0) + 1
    if answer_Index == 6:
        answer_Index = 0
    # example_output - 2
    return answer_Index, np_list
    # example_output_final - (2, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


Gender = float(input())
Height = float(input())
Weight = float(input())

# Данные взяты из learning_ai.py --> 18-21 строчки
max_Height = 199
min_Height = 140
max_Weight = 160
min_Weight = 50

# Формула X_norm = (X-min)/(max-min)
encode_inputs = [[Gender, (Height - max_Height) / (max_Height - min_Height),
                  (Weight - max_Weight) / (max_Weight - min_Weight)]]
np_encode_inputs = np.array([x for x in encode_inputs])

print(encode_inputs)

"""model = Sequential()

model.add(Dense(16, input_dim=3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights("weights.h5")

y_pred = model.predict(np_encode_inputs)
print(reverse_encode(y_pred[0]))
print(y_pred[0])
"""