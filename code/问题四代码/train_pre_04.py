from PIL import Image, ImageFilter
import os

def adjust_grayscale(image_path, threshold, op_path):
    # 打开图片
    img = Image.open(image_path)

    # 将图片转换为灰度图像
    img = img.convert('L')

    # 获取图片的像素值
    pixels = img.load()

    # 遍历每个像素,将灰度高于阈值的像素变为全白,高于阈值的像素灰度不变
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixels[x, y] > threshold:
                pixels[x, y] = 255

    # 保存调整后的图片
    img.save(os.path.join(op_path, os.path.basename(image_path)))


if __name__ == '__main__':
    image_path = r"E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\char_recog\\images"
    threshold = 170
    op_path = r"E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\char_recog\\images"
    os.makedirs(op_path, exist_ok=True)

    for _path in os.listdir(image_path):
        print(r"\n_path")
        for _ in os.listdir(os.path.join(image_path, _path)):
            try:
                adjust_grayscale(os.path.join(image_path, _path, _), threshold, os.path.join(image_path, _path))
                print(f"\r{_} is adjusted.", end="")
            except:
                print(f"{_} is failed.")

            

    print("Done.")