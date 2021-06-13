#Yiğit Kaleli - 2152007
#Yusuf Mert Köseoğlu - 2152072
#Ziya Taner Keçeci - 2152049


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob
import sys
import warnings
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

if not sys.warnoptions:
    warnings.simplefilter("ignore")

bin_size = 12
grid_level = 4

def grayscale_hist(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)     # turn image to gray scale

    # cv.imshow("gray",gray)

    histg = cv.calcHist([gray], [0], None, [bin_size], [0, 256])  # 6 bins histogram

    histg = histg / histg.sum()  # L1 normalization such that the total count of each histogram sums up to 1.

    # plt.plot(histg)
    # plt.show()
    # print(histg)



    hisc = []
    for a in histg:                        # histg is array inside an array therefore we changed into just an array
        hisc.append(a[0])

    print(hisc)

    return hisc



def grayscale_hist_grid(image):
    image = cv.resize(image, (400, 400))                         # resize image to 400x400 pixels to be able to grid into 2x2/4x4 equally
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    M = image.shape[0] // grid_level  # width of the image divided by grid level (2 = 2x2)
    N = image.shape[1] // grid_level  # length of the image

    tiles = [gray[x:x + M, y:y + N] for x in range(0, gray.shape[0], M) for y in range(0, gray.shape[1], N)]  # we are dividing our image to tiles

    # cv.imshow("gray",gray)

    histT = []                                # temporary histogram
    for i in range(len(tiles)):
        histg = cv.calcHist([gray], [0], None, [bin_size], [0, 256])  # 6 bins histogram
        his = []
        for a in histg:
            his.append(a[0])                   # histg is array inside an array therefore we changed into just an array
        histT.append(his)                      #

    histC= []
    for i in histT:
        for x in i:
            histC.append(x)


    histC = np.asarray(histC)                  # change into an asarray to be able to normalize the image


    histC = histC / histC.sum()                # L1 normalization such that the total count of each histogram sums up to 1.

    print(histC)
    # plt.plot(histg)
    # plt.show()


    return histC


def RGB_hist(image):

    histB = cv.calcHist([image], [0], None, [bin_size], [0, 256])  # compute the histogram

    histB = [val[0] for val in histB]  # convert histogram to list

    histG = cv.calcHist([image], [1], None, [bin_size], [0, 256])

    histG = [val[0] for val in histG]

    histR = cv.calcHist([image], [2], None, [bin_size], [0, 256])

    histR = [val[0] for val in histR]

    new_hist = []

    for hisR in histR:
        for hisG in histG:
            for hisB in histB:
                new_hist.append(hisR + hisG + hisB)  # pow(bins,3)

    histT = np.asarray(new_hist)  # change it to array

    histT = histT / histT.sum()  # L1 normalization

    print(histT)

    # plt.plot(histT)
    # plt.show()

    return histT


def RGB_hist_grid(image):
    image = cv.resize(image, (400, 400))


    M = image.shape[0] // grid_level
    N = image.shape[1] // grid_level

    tiles = [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]


    for i in range(len(tiles)):
        new_hist = []

        histB = cv.calcHist([tiles[i]], [0], None, [bin_size], [0, 256])  # compute the histogram

        histB = [val[0] for val in histB]  # convert histogram to list

        histG = cv.calcHist([tiles[i]], [1], None, [bin_size], [0, 256])

        histG = [val[0] for val in histG]

        histR = cv.calcHist([tiles[i]], [2], None, [bin_size], [0, 256])

        histR = [val[0] for val in histR]

        for hisR in histR:
            for hisG in histG:
                for hisB in histB:
                    new_hist.append(hisR + hisG + hisB)  # pow(bins,3)


    histT = np.asarray(new_hist)  # change it to array

    histT = histT / histT.sum()  # L1 normalization

    print(histT)

    # plt.plot(histT)
    # plt.show()

    return histT


CATEGORIES = ["cloudy", "shine", "sunrise"]

path = os.path.join(os.getcwd() + "\Dataset")
print(path)


files = glob.glob(os.path.join(os.getcwd() + "\Dataset\*"))

clouds = []
shines = []
sunrises = []

for file_name in files:                                    # we are dividing our dataset to categories of cloudy, shine and sunrise
    if ((os.path.basename(file_name).startswith("cloudy"))):
        class_num = CATEGORIES.index(CATEGORIES[0])        # decide which category it is
        cloudy = cv.imread(os.path.join(path, file_name))  # get the image
        hist_grayc = RGB_hist(cloudy)                      # get the histogram value
        clouds.append([hist_grayc, class_num])             # add it to the array
    elif ((os.path.basename(file_name).startswith("shine"))):  # Checks only base files in the data
        class_num = CATEGORIES.index(CATEGORIES[1])
        shine = cv.imread(os.path.join(path, file_name))
        hist_grays = RGB_hist(shine)  # shine 131 is problematic
        shines.append([hist_grays, class_num])
    elif ((os.path.basename(file_name).startswith("sunrise"))):  # Checks only base files in the data
        class_num = CATEGORIES.index(CATEGORIES[2])
        sunrise = cv.imread(os.path.join(path, file_name))
        hist_graysu = RGB_hist(sunrise)
        sunrises.append([hist_graysu, class_num])

divide = np.array_split(clouds, 2)       # divide the cloudy array into 2
train_cloud = divide[0];                 # create the train dataset for cloudy
divide = np.array_split(divide[1], 2)    # divide the rest to half, which becomes %25 of the cloudy dataset
validate_cloud = divide[0];              # create the validate dataset for cloudy
test_cloud = divide[1];                  # create the test dataset for cloudy

divide = np.array_split(shines, 2)
train_shine = divide[0];
divide = np.array_split(divide[1], 2)
validate_shine = divide[0];
test_shine = divide[1];

divide = np.array_split(sunrises, 2)
train_sunrise = divide[0];
divide = np.array_split(divide[1], 2)
validate_sunrise = divide[0];
test_sunrise = divide[1];

training = []                      # create the total training sets by combining all three training datasets
training.extend(train_sunrise)
training.extend(train_shine)
training.extend(train_cloud)

validate = []                      # create the total validate sets by combining all three training datasets
validate.extend(validate_sunrise)
validate.extend(validate_shine)
validate.extend(validate_cloud)

test = []                          # create the total test sets by combining all three training datasets
test.extend(test_sunrise)
test.extend(test_shine)
test.extend(test_cloud)


print("Training size", len(training))
print("Validate size", len(validate))
print("Test size", len(test))

# ---------------------------------- Training
X_train = []  # features
y_train = []  # labels

random.shuffle(training)
for features, label in training:  # spliting the training data-set with features and labels
    X_train.append(features)
    y_train.append(label)

print(len(X_train))
print(len(y_train))
# ---------------------------------- Validation
X_valid = []  # features
y_valid = []  # labels

random.shuffle(validate)
for features, label in validate:  # splitting the training data-set with features and labels
    X_valid.append(features)
    y_valid.append(label)

# ---------------------------------- Test
X_test = []  # features
y_test = []  # labels

random.shuffle(test)
for features, label in test:  # splitting the training data-set with features and labels
    X_test.append(features)
    y_test.append(label)


knn = KNeighborsClassifier(n_neighbors=5, p=2)  # Change it to 1-5-9  p=2 represents (KNN with euclidian distance)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

print(confusion_matrix(y_test, prediction))        # Show the confusion matrix
print(classification_report(y_test, prediction))   # Show the accuracy


cv.waitKey(0)

cv.destroyAllWindows()