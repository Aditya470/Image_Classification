#importing libraries and data

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
%matplotlib inline
data=input_data.read_data_sets('data/fashion',one_hot=True)
xtr=data.train.images
ytr=data.train.labels
xte=data.test.images
yte=data.test.labels
print(xtr.shape)
print(ytr.shape)
print(xte.shape)
print(yte.shape)
label_dict={0:'T-shirt/Top', 
            1:'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot',
           }
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
X_train=np.reshape(xtr,(-1,28,28,1))
X_test=np.reshape(xte,(-1,28,28,1))
Y_train=ytr
Y_test=yte
training_iters=200
learning_rate=0.001
batch_size=128
hold_prob=tf.placeholder(tf.float32)
n_input=28
n_classes=10
x=tf.placeholder("float",[None,28,28,1])
y=tf.placeholder("float",[None,n_classes])
def conv2d(x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1],padding="SAME")
weights = {
    'wc1': tf.get_variable('W1', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W3', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W4', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W5', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B5', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}
def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
pred = conv_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_loss=[]
    test_loss=[]
    train_accuracy=[]
    test_accuracy=[]
    summary_writer=tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for j in range(len(X_train)//batch_size):
            batch_x=X_train[j*batch_size:min((j+1)*batch_size,len(X_train))]
            batch_y=Y_train[j*batch_size:min((j+1)*batch_size,len(Y_train))]
            opt=sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})
            loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y})
        print('iter'+str(i)+"loss="+"{:.6f}".format(loss)+"Accuracy="+"{:.5f}".format(acc))
        print("Optimization Finished")
        test_acc,valid_loss=sess.run([accuracy,cost],feed_dict={x:X_test, y:Y_test})
        train_accuracy.append(acc)
        train_loss.append(loss)
        test_accuracy.append(test_acc)
        test_loss.append(valid_loss)
        print("test accuracy="+"{:.6f}".format(test_acc))    
    summary_writer.close()

plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()
