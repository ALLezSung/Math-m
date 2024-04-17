import os
import shutil


def main(raw_path, op_path, folder_name, sym, max_file_nums):
    file_num = 0
    os.makedirs(os.path.join(op_path, 'images', folder_name), exist_ok=True)
    for image in os.listdir(raw_path + '\\' + folder_name):    
        shutil.copyfile(os.path.join(raw_path, folder_name, image), os.path.join(op_path, 'images', folder_name, image))
        with open(op_path + '\\labels\\' + image.split(".")[0] + '.txt', "w") as file:
            file.write(f'{sym} 0.5 0.5 1 1\n')
        file_num += 1
        if file_num >= max_file_nums:
            break

if __name__ == '__main__':
    raw_path = r'C:\\Users\\iDo\\Desktop\\4_Recognize\\训练集beta'
    op_path = r'E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\char_recog'
    max_file_nums = 200
    
    os.makedirs(os.path.join(op_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(op_path, 'labels'), exist_ok=True)

    sym = 0
    for folder_name in os.listdir(raw_path):
        main(raw_path, op_path, folder_name, sym, max_file_nums)
        sym += 1

    print('Done.')