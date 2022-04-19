import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
labels = np.uint8(np.loadtxt("mid/ans.txt"))
# 讀入 test_dataset.avi
vidcap = cv2.VideoCapture("test_dataset.avi")

success = True

# 存放幀的 list
video_frame = []

while success:
    # 讀入幀
    success, image = vidcap.read()
    print(success)
    # 確認還有沒有幀
    if not success:
        break

    # 將影像由 RGB 轉成 GRAY（這樣只會保留明亮度，所以陣列只有二維，RGB 會出現三維不好處理）
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 加到 list 裡面
    video_frame.append(image)
data = np.array(video_frame).reshape((len(video_frame), -1))
    # 創建一個 plot 來放我們的預期圖片
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

# 開始遍歷 plot, data, label
for ax, image, label in zip(axes, data, labels):

    # 關閉 x 軸線（不需要 x 軸線）
    ax.set_axis_off()

    # 將圖片從剛剛轉換的資料，轉回來二維的圖片
    image = image.reshape(28, 28)

    # 顯示圖片
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")

    # 設定標題
    ax.set_title("Training: %i" % label)
    
    # 存圖片
    ax.get_figure().savefig("output1.png")
# 創建一個 K-neighbor classifier 分類器
clf = KNeighborsClassifier()

# 將資料及分割成 1% 的測試資料與 99% 的訓練資料
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)

# 訓練
clf.fit(x_train, y_train)

# 預測
predicted = clf.predict(x_test)
# 用來遍歷我們的預測資料
index = 0

# 創建一個 plot
_, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

# 因為有四格，所以先從欄開始
for axs in axes:
    
    # 再從列開始
    for ax in axs:

        # 不需要 x 軸
        ax.set_axis_off()

        # 把圖片轉回去 (28, 28) 的形式
        sk_image = x_test[index].reshape(28, 28)

        # 讀入預測結果
        prediction = predicted[index]

        # 顯示圖片
        ax.imshow(sk_image, cmap=plt.cm.gray_r, interpolation="nearest")
        
        # 設置標題
        ax.set_title(f"Prediction: {prediction}")
        
        # 存成檔案
        ax.get_figure().savefig("output2.png")

        # 繼續遍歷，所以 index += 1
        index += 1
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)

# 設置標題
disp.figure_.suptitle("Confusion Matrix")

# 顯示出文字版的 confusion matrix
print(f"Confusion matrix:\n{disp.confusion_matrix}")

# 顯示出圖片版的 confusion matrix
plt.show()

# 存成圖片
plt.savefig("output3.png")