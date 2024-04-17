import os
import cv2
import json
import shutil

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root, f)
                yield fullname

def main(base, result):
    # 确保结果路径中的images和labels文件夹存在
    os.makedirs(os.path.join(result, 'images'), exist_ok=True)
    os.makedirs(os.path.join(result, 'labels'), exist_ok=True)
    
    for json_file in findAllFile(base):
        try:
            json_data = json.load(open(json_file))
            
            img_name = json_data["img_name"] + '.jpg'
            image_path = os.path.join(base, json_data["img_name"] + '.jpg')
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            line = ""
            for obj in json_data['ann']:          
                if obj[-1]:
                    x1 = obj[0] / width
                    y1 = obj[1] / height
                    x2 = obj[2] / width
                    y2 = obj[3] / height

                    cx = round((x1 + x2) / 2, 6)
                    cy = round((y1 + y2) / 2, 6)
                    wid = round(x2 - x1, 6)
                    hei = round(y2 - y1, 6)
                    
                    line += f'0 {cx} {cy} {wid} {hei}\n'
                else:
                    pass
            
            # 构建目标图像的保存路径
            img_save_path = os.path.join(result, 'images', img_name)
            # 复制图像文件到结果路径的images文件夹
            shutil.copyfile(image_path, img_save_path)
            
            with open(os.path.join(result, 'labels', json_data["img_name"] + '.txt'), 'w') as f:
                f.write(line)
        except:
            pass

if __name__ == '__main__':
    data_path = 'C:\\Users\\iDo\\Desktop\\2_Train'
    op_path = 'E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\my_dss'
    main(data_path, op_path)
    print("Done.")