import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2

from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist


# Step 0: Define the neural network model, return logits instead of activation in forward method

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


# Step 1: Load the MNIST dataset

(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

print(x_test.shape, y_test.shape)

x_test = x_test[:500,:,:,:]
y_test = y_test[:500,:]

# Step 1a: Swap axes to PyTorch's NCHW format

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# Step 2: Create the model

model = Net()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

# Step 4: Train the ART classifier

classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)

accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = AdversarialPatch(classifier=classifier,patch_shape=(1,3,3),targeted=False)
patch= attack.generate(x=x_test)
x_test_adv = attack.apply_patch(x=x_test,scale=1,patch_external=patch)


print(type(x_test), x_test.shape)
print(type(x_test_adv),x_test_adv.shape)

# Display the images

#x1 = x_test[0][0]
#x2 = x_test_adv[0][0]
#x1 = cv2.resize(x1,dsize=(140,140))
#x2 = cv2.resize(x2,dsize=(140,140))
#print(x1.shape)
#cv2.imshow("test",x1)
#cv2.waitKey(0)
#cv2.imshow("test_adv",x2)
#cv2.waitKey(0)


# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(np.array(x_test_adv))
print(np.argmax(predictions,axis=1))
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))