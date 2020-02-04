import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
(X_train,Y_train), (X_test, Y_test) = mnist.load_data()
epochs = 10
batch_size = 100
X_train = X_train/255.0
X_test = X_test/255.0
X_test = tf.Variable(X_test)

w1 = tf.Variable(tf.random.normal([784,300],stddev=0.03),name="w1")
b1 = tf.Variable(tf.random.normal([300]),name="b1")

w2 = tf.Variable(tf.random.normal([300,10],stddev = 0.03),name="w2")
b2 = tf.Variable(tf.random.normal([10]),name="b2")

def nn_model(x_input,w1,b1,w2,b2):
    x_input = tf.reshape(x_input, (x_input.shape[0],-1))#w1 784x300 b1 = 300x1
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32),w1),b1)#x = 100x300 w2 = 300x10 b2 = 10x1
    x = tf.nn.relu(x)
    logits = tf.add(tf.matmul(x,w2),b2)
    return logits #100x10

def get_batch(x_data,y_data,batch_size):
    indx = np.random.randint(0,len(y_data),batch_size)
    return x_data[indx,:,:], y_data[indx]

def lossfn(logits,labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))

optimizer = tf.keras.optimizers.Adam()

for epoch in range(epochs):
    avg_loss = 0
    total_batches = int(len(Y_train)/batch_size)
    for i in range(total_batches):
        batch_x,batch_y = get_batch(X_train,Y_train,batch_size)
        batch_x = tf.Variable(batch_x)
        batch_y = tf.Variable(batch_y)
        batch_y = tf.one_hot(batch_y,10)
        with tf.GradientTape() as tape:
            logits = nn_model(batch_x,w1,b1,w2,b2)
            loss = lossfn(logits,batch_y)
        gradients = tape.gradient(loss,[w1,b1,w2,b2])
        optimizer.apply_gradients(zip(gradients,[w1,b1,w2,b2]))
        avg_loss += loss/total_batches

    test_logits = nn_model(X_test,w1,b1,w2,b2)
    max_index = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_index.numpy() == Y_test)/len(Y_test)

print("Training complete!")