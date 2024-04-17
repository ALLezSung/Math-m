from ultralytics import YOLO

if __name__ == '__main__':
    yolo_path = r'C:\\Users\\iDo\\Desktop\\py\\yolov8m.pt'
    yaml_path = r'E:\\.app\\GitsDepository\\Gitshub-projects\\ultralytics\\ultralytics\\cfg\\models\\v8\\yolov8m.yaml'
    data_path = r'E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\char_recog_split\\data.yaml'

    # Load a model
    model = YOLO(yaml_path) # build a new model from YAML
    model = YOLO(yolo_path)  # load a pretrained model

    # recommended for training
    model = YOLO(yaml_path).load(yolo_path)  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=data_path,
                          epochs=100, patience=20, 
                          imgsz=224, batch=64, 
                          plots=True)