import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

np.random.seed(42)
df1 = pd.read_csv('../dataset/APUF_XOR_Challenge_Parity_64_500000.csv', header=None)
df2 = pd.read_csv('../all_apuf_responses/1resp_XOR_APUF_chal_64_500000.csv', header=None)

X = df1.iloc[:8499, :65]
Y = df2.iloc[:8499, :]
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)

model = keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='relu'))  # layer 1: 5 output
model.add(layers.Dense(1, activation='sigmoid'))  # layer 2 output
# another coding way
# model = keras.Sequential([
#     layers.Dense(5, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
model.build(input_shape=[None, 65])
# summary() method is same as print()
model.summary()
# draw a model flow graph
# keras.utils.plot_model(model, 'model.png')
# keras.utils.plot_model(model, 'model_info.png', show_shapes=True)
for p in model.trainable_variables:
    print(p.name, p.shape)

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
# 1000个72维数据

model.fit(train_features, train_labels,
          epochs=500,
          batch_size=1000,
          validation_data=(test_features, test_labels)
          )

scores = model.evaluate(test_features, test_labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
