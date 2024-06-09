from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



label_map = {1: 'fake', 0: 'real'}


df = pd.read_csv('predictions_test3.csv')

df['label'] = df['label'].str.lower()
df['predicted_class'] = df['predicted_class'].str.lower()
print("ASIL DATAA", df['predicted_class'].where(df['predicted_class'] == 'false').count())

fake_fake = 0
fake_real = 0
real_fake = 0
real_real = 0


for i in range(len(df)):
    if df['label'][i] == 'real' and df['predicted_class'][i] == 'fake':
        fake_fake += 1
    elif df['label'][i] == 'real' and df['predicted_class'][i] == 'real':
        fake_real += 1
    elif df['label'][i] == 'real' and df['predicted_class'][i] == 'fake':
        real_fake += 1
    elif df['label'][i] == 'real' and df['predicted_class'][i] == 'real':
        real_real += 1

print(f'fake_fake: {fake_fake}')
print(f'fake_real: {fake_real}')
print(f'real_fake: {real_fake}')
print(f'real_real: {real_real}')




# Define true labels (ground truth) from your dataset
true_labels = df['label'].tolist()  # Replace 'true_label_column' with the column name containing the true labels

pred = df['predicted_class'].tolist()

# Create confusion matrix
cm = confusion_matrix(true_labels, pred)

# Plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' )
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
