from sklearn import tree  # decision tree model
from sklearn.datasets import load_digits  # import dataset
from sklearn.model_selection import train_test_split  # splitting our data
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

data = load_digits()  # load digit dataset
class_names = data.target_names
print(dir(data))  # show dataset meta

# prepare train and test dataset
x_train, x_test, y_train, y_test = \
    train_test_split(data.data, data.target, test_size=0.20, random_state=42)

# show data size -->
print("Training set" + str(x_train.shape))
print("Test set " + str(x_test.shape))
print("Classes for training set" + str(y_train.shape))
print("Classes for test set" + str(y_test.shape))

# classifier object
c45_classifier = tree.DecisionTreeClassifier(criterion='entropy')
# c45_classifier.fit(x_train, y_train)

# train and test the model
y_pred = c45_classifier.fit(x_train, y_train).predict(x_test)

# calculate train accuracy
train_accuracy = (np.sum(c45_classifier.predict(x_train) == y_train) / float(y_train.size)) * 100

# calculate test accuracy
test_accuracy = (np.sum(c45_classifier.predict(x_test) == y_test) / float(y_test.size)) * 100

# print accuracies
print("Model accuray on train sample ", str(train_accuracy) + "%")
print("Model accuray on test sample", str(test_accuracy) + "%")

# precision, recall, f1-score, support
print(classification_report(y_test, y_pred))


# export our decision tree
# with open('digit_classifier.txt', 'w') as export_file:  # create file object
#     # write c45_classifier info to export_file using export_graphviz
#     tree.export_graphviz(c45_classifier,
#                          out_file=export_file,
#                          feature_names=data.data,
#                          class_names=data.target_names,
#                          filled=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='confusion matrix for Digits Data Set')

# show confusion matrix figures
plt.show()
