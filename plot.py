import matplotlib.pyplot as plt

# Data
x = [0.1, 0.2, 0.5, 1]
y = [62.6, 62.6, 62.6, 62.6]
y2 = [51.5, 56.9, 61.7, 64.7]
y3 = [67.6, 68.5, 70.3, 71.6]
y4 = [73.7, 75, 76.6, 77.5]

# Plotting
plt.figure(figsize=(8, 5))

l=3
# Plotting the line
plt.plot(x, y, color='yellowgreen', label='TableQA', linestyle='dashed', linewidth=l)
plt.plot(x, y2, color='darkgreen', label='Text-to-SQL', marker='o', linewidth=l)
plt.plot(x, y3, color='darkorchid', label='SynTQA(RF)', marker='o', linewidth=l)
plt.plot(x, y4, color='blue', label='Oracle', marker='o', linewidth=l)


# Customizing the plot
plt.xlabel('SQL Annotation Amount', fontsize=20,labelpad=10)
plt.ylabel('Accuracy', fontsize=20, labelpad=10)
plt.xticks(x, ['10%', '20%', '50%', '100%'], fontsize=16)
plt.yticks(range(50, 81, 5), [f'{val}%' for val in range(50, 81, 5)], fontsize=16)
plt.grid(True, linestyle='-', color='lightgrey')

#
plt.legend(loc='lower right', fontsize=16, ncol=2) 

# plt.legend(loc='upper center', bbox_to_anchor=(0.46, 1.35),
#           fancybox=True, shadow=False, ncol=2, fontsize=16)

plt.tight_layout()
plt.savefig('line_plot.pdf')




