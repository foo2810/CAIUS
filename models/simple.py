import tensorflow as tf
tfk = tf.keras

class SimpleCNN(tfk.Model):
    def __init__(self, n_out=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = tfk.layers.Conv2D(32, 3, 1, 'same')
        self.conv2 = tfk.layers.Conv2D(64, 3, 1, 'same')
        self.conv3 = tfk.layers.Conv2D(128, 3, 1, 'same')

        self.pool1 = tfk.layers.MaxPool2D(2, 2)
        self.pool2 = tfk.layers.MaxPool2D(2, 2)
        self.pool3 = tfk.layers.MaxPool2D(2, 2)

        self.relu = tfk.layers.ReLU()

        self.fc1 = tfk.layers.Dense(1024)
        self.fc2= tfk.layers.Dense(1024)
        self.fc3= tfk.layers.Dense(n_out)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = tfk.activations.softmax(x)
        return x