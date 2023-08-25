from keras.layers import Dense
from keras.models import Sequential
import numpy as np


def AI(Gender, Height, Weight):
    def reverse_encode(np_list):
        # example_input - [1.4149497e-01 8.8076782e-01 2.3201625e-01 4.9572612e-01 5.8198853e-01
        #  1.3676369e-01]
        np_list = list(np_list)

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
        return answer_Index
        # example_output_final - (2, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # Данные взяты из learning_ai.py --> 18-21 строчки
    max_Height = 199
    min_Height = 140
    max_Weight = 160
    min_Weight = 50

    # Формула X_norm = (X-min)/(max-min)
    encode_inputs = [[Gender, (Height - max_Height) / (max_Height - min_Height),
                      (Weight - max_Weight) / (max_Weight - min_Weight)]]
    np_encode_inputs = np.array([x for x in encode_inputs])

    model = Sequential()

    model.add(Dense(15, input_dim=3, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights("files/weights.h5")

    y_pred = model.predict(np_encode_inputs)
    return reverse_encode(y_pred[0])

"""
1.Рост-177,вес-58,пол-женский - 2
2.рост-174,вес-69,пол-мужской - 2
3.Рост-182,вес-64,пол-мужской -2
4.Рост-177,вес-59,пол-женский -2
5.Рост-183,вес-60,пол-мужской -2
6.Рост-175,вес-60,пол-женский -2
7.Рост-181,вес-75,пол-мужской -2
8.Рост-174,вес-60,пол-женский -2
9.Рост-173,вес-55,пол-женский -2
10.Рост-188,вес-79,пол-мужской -2
11.Рост-165,вес-58,пол-женский -2
12.Рост-185,вес-75,пол-мужской -2
13.Рост-175,вес-56,пол-женский -2
14.Рост-190,вес-80,пол-мужской -2
15.Рост-184,вес-71,пол-мужской -2
16.Рост-178,вес-59,пол-женский -2
"""

