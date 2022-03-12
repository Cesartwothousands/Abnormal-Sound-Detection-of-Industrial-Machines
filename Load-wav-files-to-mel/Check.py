import os

path = r'F:\毕业论文\Industrial-Machine-Investigation-and-Inspection\Load-wav-files-to-mel'
files = os.listdir(path)
Files = []

for file in files:
    if file[-3:] == 'npy':
        Files.append(file)

num_npy = len(Files)
print(num_npy)