import os
import random
import shutil

def move_files(data_path, op_path, train_scale, valid_scale, img_folder_name):
    # 定义图片和标签的文件夹路径
    image_folder = os.path.join(data_path, 'images', img_folder_name)
    label_folder = os.path.join(data_path, 'labels')

    # 获取所有图片的文件名（不含扩展名）
    images = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, file))]
    # 打乱文件顺序
    random.shuffle(images)

    # 计算训练、验证和测试集各自所需的图片数量
    num_train = int(len(images) * train_scale)
    num_valid = int(len(images) * valid_scale)
    # 测试集的数量由总数减去训练和验证集的数量得到
    num_test = len(images) - num_train - num_valid

    # 分配文件到训练、验证和测试集
    train_images = images[:num_train]
    valid_images = images[num_train:num_train+num_valid]
    test_images = images[num_train+num_valid:]

    # 创建训练、验证和测试的目标文件夹
    for phase in [('train', train_images), ('valid', valid_images), ('test', test_images)]:
        phase_name, phase_images = phase
        os.makedirs(os.path.join(op_path, phase_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(op_path, phase_name, 'labels'), exist_ok=True)

        # 移动每张图片及其对应的标签
        for image_name in phase_images:
            try:
                src_image_path = os.path.join(image_folder, image_name + '.bmp')
                dst_image_path = os.path.join(op_path, phase_name, 'images', image_name + '.bmp')
                shutil.copy2(src_image_path, dst_image_path)
            except FileNotFoundError:
                src_image_path = os.path.join(image_folder, image_name + '.png')
                dst_image_path = os.path.join(op_path, phase_name, 'images', image_name + '.png')
                shutil.copy2(src_image_path, dst_image_path)

            # 根据图片文件名转移对应的标签文件
            src_label_path = os.path.join(label_folder, image_name + '.txt')
            dst_label_path = os.path.join(op_path, phase_name, 'labels', image_name + '.txt')
            # 只有当标签文件存在时才移动它
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)

if __name__ == "__main__":
    data_path = "E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\char_recog"
    op_path = "E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\char_recog_split"
    train_scale = 0.8
    valid_scale = 0.2
    test_scale = 0.0
    img_folder_list = os.listdir(data_path + "\\images") 


    for img_folder_name in img_folder_list:
        move_files(data_path, op_path, train_scale, valid_scale, img_folder_name)
    print("Done.")