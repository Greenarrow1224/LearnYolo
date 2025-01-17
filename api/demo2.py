from ultralytics import YOLO
import cv2
def test():
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='./conf/VisDrone.yaml', epochs=30)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model('./Magic.jpg')
    res = results[0].plot()
    cv2.imshow("YOLOv8 Detection", res)
    cv2.waitKey(10)
    # Export the model to ONNX format
    # success = model.export(format='onnx')
if __name__ == '__main__':
    test()