import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot  as plt
from re import X
mnist = tf.keras.datasets.mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_test, X_train = X_test / 255.0, X_train / 255.0

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,validation_data = (X_test,y_test))
test_loss,test_acc = model.evaluate(X_test,y_test)
print(f"Test accuracy : {test_acc : .4f}")

predictons = model.predict(X_test)

plt.imshow(X_test[1].reshape(28,28),cmap = "gray")
plt.title(f"Predicted Label: {predictons[1].argmax()}")
plt.show()