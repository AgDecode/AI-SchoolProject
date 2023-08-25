import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
import numpy as np

dataset = pd.read_csv("D:/code_(python)/Machine Learning school-project/test_ai/data_for_ai/network_training (500_Person_Gender_Height_Weight_Index).csv")

# Переписываем функцию OneHotEncoder и StandardScaler "не черный ящик"
# ---------------------------------------------------------------------------------------------------------
input_names = ["Gender", "Height", "Weight"]
output_names = ["Index"]

dataset_test_input = dataset[input_names][0:1]
print(dataset_test_input)

max_Height = dataset["Height"].max()
min_Height = dataset["Height"].min()
max_Weight = dataset["Weight"].max()
min_Weight = dataset["Weight"].min()

encoders = {"Height": lambda Height: [(Height - max_Height) / (max_Height - min_Height)],
            "Weight": lambda Weight: [(Weight - max_Weight) / (max_Weight - min_Weight)],
            "Index": lambda Index: {0: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    2: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 3: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                    4: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 5: [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]}.get(Index),
            "Gender": lambda s_Gender: [s_Gender]}


def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = dataset[column].values
        result[column] = values
    return result


def make_supervised():
    raw_input_data = dataset[input_names]
    raw_output_data = dataset[output_names]
    return {"inputs": dataframe_to_dict(raw_input_data),
            "outputs": dataframe_to_dict(raw_output_data), }


def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded = list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formatted = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formatted.append(vector)
    return formatted


def reverse_encode(np_list):
    np_list = list(np_list)
    max_number_index = np_list.index(max(np_list))
    np_list[max_number_index] = 1.0
    for i in range(len(np_list)):
        if np_list[i] != 1.0:
            np_list[i] = 0.0
    return np_list


supervised = make_supervised()
encode_inputs = encode(supervised["inputs"])
encode_outputs = encode(supervised["outputs"])

np_encode_inputs = np.array([x for x in encode_inputs])
np_encode_outputs = np.array([x for x in encode_outputs])

# Переписываем функцию train_test_split
x_train = np_encode_inputs[:125]
y_train = np_encode_outputs[:125]

x_test = np_encode_inputs[125:]
y_test = np_encode_outputs[125:]
# ---------------------------------------------------------------------------------------------------------


model = Sequential()

model.add(Dense(16, input_dim=3, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

y_pred = model.predict(x_test)

print(x_test[0])
real_data = dataset.iloc[125:][input_names + output_names]
print(reverse_encode(y_pred[0]))

# model.save_weights("weights.h5")
