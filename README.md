# DD2424-Project

Knowlegde distillation by teacher-student model. Using Yolov3 model for teacher and Yolov3-tiny model for student. Both models are initially initilized on pre-trained weights.
The Yolov3model is fine tuned on PASCAL and aftwards distillation processs is used for Yolov3-tiny model.

Fine-tuned weights can be found here https://drive.google.com/drive/folders/1gZUBQiYLDPtXC20w9ICyXCMnup03EjkN?usp=sharing

This repo uses the work of https://github.com/qqwweee/keras-yolo3

1. You need the weights to yolov3 and yolov3-tiny
2. Convert Darknets .weights to Keras .h5
3. Load up pre-trained .5 for yolov3 and fine-tune model
4. Load up fine-tuned .5  yolov3

5. Load up pre-trained .5 yolov3-tiny and fine-tune model
6. Create Distillation model
7. Train Distillation Model

Alteration in :
  Class YOLO:

   generate() : added by_name = True in self.yolo_model = load_model(model_path, compile=False, by_name = True)

   predict(): added a new method partially based on detect_image()

  train.py
    test_generator(): added a new generator partially based on data_generator()
    PIL Image : imported PIL
