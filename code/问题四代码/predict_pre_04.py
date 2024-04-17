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
    image_path = r"C:\\Users\\iDo\\Desktop\\4_Recognize\\测试集"
    threshold = 170
    op_path = r"C:\\Users\\iDo\\Desktop\\4_Recognize\\测试集"
    os.makedirs(op_path, exist_ok=True)

    for _ in os.listdir(image_path):
        if _.endswith(".jpg"):
            try:
                adjust_grayscale(os.path.join(image_path, _), threshold, op_path)
                print(f"\r{_} is adjusted.", end="")
            except:
                pass
        

    print("Done.")