from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import re


def create_folders(_path):
    folder_path = _path + "\\all"
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            if file.endswith(".jpg"):
                # 获取文件名
                file_name = os.path.splitext(file)[0]
                # 获取文件所在的文件夹
                file_folder = os.path.dirname(os.path.join(_path, file))
                # 在上一级文件夹中创建同名的文件夹
                new_folder_path = os.path.join(file_folder, file_name)
                os.makedirs(new_folder_path, exist_ok=True)
                # 将对应的.jpg文件复制一份到同名文件夹中
                src = os.path.join(folder_path, file)
                dst = os.path.join(new_folder_path, file)
                shutil.copy2(src, dst)

def fetch_images(_path, images_path):
    # 大文件夹和图像存储文件夹的路径，请根据实际情况替换这里的路径
    parent_folder_path = _path
    image_folder_path = images_path

    # 遍历大文件夹中的所有小文件夹
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(folder_path):
            # 构建符合条件的文件名正则表达式
            search_pattern = re.compile(rf"^{re.escape(folder_name)}(\d+)?\.jpg$", re.I)

            # 在图像文件夹中搜索符合条件的图像
            _ = 1
            for image_name in os.listdir(image_folder_path):
                if search_pattern.match(image_name):
                    image_path = os.path.join(image_folder_path, image_name)
                    try:
                        conf_path = os.path.join(output_folder_path, "all\\pred\\labels", f"{image_name.split('.')[0]}.txt")
                        with open(conf_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                odj_label = label_list[eval(lines[0].split(' ')[0])]
                    except:
                        odj_label = "none"
                    target_path = os.path.join(folder_path, f"{_}-{odj_label}" + ".jpg")

                    # 复制图像到对应的小文件夹
                    shutil.copy2(image_path, target_path)
                    _ += 1

label_list = ['万', '丘', '丙', '丧', '乘', '亦', '人', '今', '介', '从', '令', '以', '伊', '何', '余', '允', '元', '兄', '光', '兔',
                '入', '凤', '化', '北', '印', '及', '取', '口', '吉', '囚', '夫', '央', '宗', '宾', '尞', '巳', '帽', '并', '彘', '往', 
                '御', '微', '旨', '昃', '木', '朿', '涎', '灾', '焦', '爽', '牝', '牡', '牧', '生', '田', '疑', '祝', '福', '立', '羊', 
                '羌', '翌', '翼', '老', '艰', '艺', '若', '莫', '获', '衣', '逆', '门', '降', '陟', '雍', '鹿']


if __name__ == "__main__":
    
    # 路径
    model_a_path = "C:\\Users\\iDo\\Desktop\\预处理后\\weights\\best.pt"
    model_b_path = "C:\\Users\\iDo\\Desktop\\预处理后_04\\weights\\best.pt"
    image_folder_path = "C:\\Users\\iDo\\Desktop\\4_Recognize\\测试集"
    output_folder_path = "C:\\Users\\iDo\\Desktop\\4_Recognize\\results"
    os.makedirs(output_folder_path, exist_ok=True)

    # 加载模型
    model_a = YOLO(model_a_path)
    model_b = YOLO(model_b_path)

    # 模型A 预测
    results_a = model_a.predict(image_folder_path, save=True, save_crop=True,save_txt=True, project=output_folder_path,show_labels=False , name="all")
    
    create_folders(output_folder_path)

    image_folder_path = output_folder_path + "\\all\\crops\\OBS"
    # 模型B 预测
    results_b = model_b.predict(image_folder_path, save=True,save_txt=True, project=output_folder_path, name="all\\pred")

    fetch_images(output_folder_path, os.path.join(output_folder_path, "all\\crops\\OBS"))