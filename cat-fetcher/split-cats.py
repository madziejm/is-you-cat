import os
import shutil

i = 0
os.mkdir('cats_1')
os.mkdir('cats_2')
os.mkdir('cats_3')
for filename in os.listdir('temp'):
    if not filename.endswith('.jpg'):
        print(f"hmm {filename}")
        continue
    filepath = os.path.join('temp', filename)
    if 0 <= i and i <= 10498:
        shutil.move(filepath, os.path.join(
            'cats_1',
            filename
            ))
    elif 10499 <= i and i <= 11498:
        shutil.move(filepath, os.path.join(
            'cats_2',
            filename
            ))
    elif 11499 <= i and i <= 12499:
        shutil.move(filepath, os.path.join(
            'cats_3',
            filename
            ))
    i+=1
