import argparse
import os
import cv2
from progressbar import ProgressBar
pbar = ProgressBar()

import warnings
warnings.filterwarnings("ignore")

def main(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)

    for each in pbar(os.listdir(src)):
        img = cv2.imread(os.path.join(src,each))
        img = cv2.resize(img,(64,64))
        cv2.imwrite(os.path.join(dst,each), img)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize Input Images')
    parser.add_argument('--input',type=str,required=True,help='Directory containing images to resize. eg: ./data')
    parser.add_argument('--output',type=str,required=True,help='Directory to save resized images. eg: ./resized')
    args = parser.parse_args()
    main(args.input, args.output)
    
