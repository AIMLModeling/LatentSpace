import argparse
from PIL import Image
import os

def main(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
        
    for each in os.listdir(src):
        png = Image.open(os.path.join(src,each))
        if png.mode == 'RGBA':
            png.load()
            background = Image.new("RGB", png.size, (0,0,0))
            background.paste(png, mask=png.split()[3])
            background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
        else:
            png.convert('RGB')
            png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert RGBA to RGB')
    parser.add_argument('--input',type=str,required=True,help='Directory containing images to resize. eg: ./resized')
    parser.add_argument('--output',type=str,required=True,help='Directory to save resized images. eg: ./RGB_data')
    args = parser.parse_args()
    main(args.input, args.output)
