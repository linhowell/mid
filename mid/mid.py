import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

ans = np.uint8(np.loadtxt("mid/ans.txt"))

vidio = cv2.VideoCapture("mid/test_dataset.avi")

isrun = True
arr = []
while isrun:
    isrun,img = vidio.read()
    if isrun == False:
        break
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        arr.append(img)

data = np.array(arr).reshape((len(arr), -1))
train = KNeighborsClassifier()
x_train, x_test, y_train, y_test = train_test_split(data, ans, test_size=0.05)
train.fit(x_train, y_train)
predicted = train.predict(x_test)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
plt.show()

