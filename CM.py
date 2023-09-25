# import numpy as np

# # Given metrics
# accuracy = 0.88
# f1_score = 0.84
# recall = 0.85
# precision = 0.84

# # Total images and class distribution 
# total = 25
# class_0 = 12
# class_1 = 13

# # Calculate true positives, false positives, etc.
# tp = round(recall * class_1) # True positives 
# fp = round(class_0 - tp) # False positives
# fn = round(class_1 - tp) # False negatives
# tn = round(total - tp - fp - fn) # True negatives

# # Create confusion matrix
# confusion_matrix = np.array([[tn, fp], [fn, tp]])

# print(confusion_matrix)


# import matplotlib.pyplot as plt

# # Confusion matrix from previous code
# confusion_matrix = np.array([[10, 2], [3, 10]])

# # Calculate true positive rate and false positive rate
# tp = confusion_matrix[1,1] 
# fp = confusion_matrix[0,1]
# fn = confusion_matrix[1,0]
# tn = confusion_matrix[0,0]

# tpr = tp / (tp + fn) # True positive rate (recall)
# fpr = fp / (fp + tn) # False positive rate

# # Plot ROC curve
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot([0, fpr], [0, tpr], label='ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()


import numpy as np

# Given metrics
# accuracy = 0.88
# f1_score = 0.84
# recall = 0.85
# precision = 0.84

# Given metrics
accuracy = 0.88
f1_score = 0.88
recall = 0.88
precision = 0.89

# Class distribution
total = 25
class_0 = 12
class_1 = 13

# Calculate true positives (TP) and false positives (FP)
tp = round(recall * class_1)
fp = round(tp / precision - tp)

# Calculate true negatives (TN) and false negatives (FN)
tn = round(class_0 - fp)
fn = round(class_1 - tp)

# Create confusion matrix
confusion_matrix = np.array([[tp, fp], [fn, tn]])

print(confusion_matrix)

