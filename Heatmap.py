<<<<<<< HEAD
import matplotlib.pyplot as plt
import seaborn as sns

data = [[1098, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1132, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1146, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1113, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1100, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1087, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1115, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1118, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1090]]

plt.figure(figsize=(9,9))
sns.heatmap(data=data, annot=True, fmt = 'd', linewidths=.5, cmap='Blues')
=======
import matplotlib.pyplot as plt
import seaborn as sns

data = [[1098, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1132, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1146, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1113, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1100, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1087, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1115, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1118, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1090]]

plt.figure(figsize=(9,9))
sns.heatmap(data=data, annot=True, fmt = 'd', linewidths=.5, cmap='Blues')
>>>>>>> 415b628f98856795908f91cbd80e1edf79075e5a
plt.show()