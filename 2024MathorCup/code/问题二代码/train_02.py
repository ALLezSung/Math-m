from ultralytics import YOLO

if __name__ == '__main__':
    yolo_path = r'C:\\Users\\iDo\\Desktop\\models\\yolov8\\yolov8n.pt'
    yaml_path = r'E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\my_dss_split\\yolov8n.yaml'
    data_path = r'E:\\.app\\GitsDepository\\Gitshub-projects\\datasets\\my_dss_split\\data.yaml'

    # Load a model
    model = YOLO(yaml_path) # build a new model from YAML
    model = YOLO(yolo_path)  # load a pretrained model
   
    # recommended for training
    model = YOLO(yaml_path).load(yolo_path)  # build from YAML and transfer weights
    
    # Train the model
    results = model.train(data=data_path,
                          epochs=100, imgsz=608, patience=20, plots=True, batch=48)