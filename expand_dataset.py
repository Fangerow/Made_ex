import cv2 as cv
import os
from tqdm.auto import tqdm

from geometry import Geometry


def main():
    image = Geometry()
    news_num = 1600
    with open(r"C:\Users\User\Downloads\train_sq.csv", 'w') as solution:
        path_fold = r'C:\Users\User\Downloads\train_data'
        for i in tqdm(range(news_num)):
            path = os.path.join(path_fold, str(3001 + i) + '.png')
            img, label = image.add_geom()
            cv.imwrite(path, img)
            solution.write(f'train_data/{3001 + i}.png,{label}' + '\n')


if __name__ == "__main__":
    main()
