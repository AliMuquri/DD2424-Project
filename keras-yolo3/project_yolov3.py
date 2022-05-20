
# !pip uninstall keras
#!pip install keras==2.1.5
# #!pip install tensorflow-gpu==1.6.0
# !pip install 'h5py==2.10.0' --force-reinstall

# Commented out IPython magic to ensure Python compatibility.
#%tensorflow_version 1.x
import tensorflow.compat.v1 as tf
import keras
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import wget

tf.compat.v1.disable_eager_execution()
#IF THE VERSION ARE NOT CORRECT RESTART RUNTIME.
#Try also restart run, reinstall
# correct versions below
print(tf.__version__) #1.15.2
print(keras.__version__)  #2.1.5
print(h5py.__version__) # 2.10.0

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

#The repo below is a popular repo that is mentioned repeatable for those who want to translate darknet models to keras models. Primarly yolov3
#!git clone https://github.com/qqwweee/keras-yolo3
#!git clone https://github.com/AliMuquri/DD2424-Project.git

# Commented out IPython magic to ensure Python compatibility.
#%cd keras-yolo3
# %cd DD2424-Project
# %cd keras-yolo3

#py version
#os.chdir("./keras-yolo3")


#1. Below chose weights to retrive
#2. If you need to convert the weight choose converter
#3. Retrieve dataset and preproccess

#!python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolov3-tiny_weights.h5

#IMPORTING all weights and PASCAL training set. The above repo included a data preprocessing script that is used "voc_annotation.py"
#The pretrained weights are retrieved form here and the dataset
#!wget https://pjreddie.com/media/files/yolov3.weights
#!wget https://pjreddie.com/media/files/yolov3-tiny.weights
#!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1S1V-V0JtbQE6Z0LQmEMNPJJyJYgjFSsY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1S1V-V0JtbQE6Z0LQmEMNPJJyJYgjFSsY" -O fine_tune_yolov3.h5 && rm -rf /tmp/cookies.txt
#!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_fAf7NGyY3-Zp0PDQwQ1gsfxn2hZdvHY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_fAf7NGyY3-Zp0PDQwQ1gsfxn2hZdvHY" -O fine_tune_yolov3-tiny.h5 && rm -rf /tmp/cookies.txt

#USING REPO convert.py to convert from darknet model to keras model and save it a uploadable .h5 file
# !python  convert.py yolov3.cfg yolov3.weights model_data/yolov3_weights.h5
#!python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolov3-tiny_weights.h5

#We need a weights only model for distiller
#!python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolov3-tiny_weights.h5 --weights_only

!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
!wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
#!tar xvf VOCtrainval_06-Nov-2007.tar
#!tar xvf VOCtest_06-Nov-2007.tar
#!python voc_annotation.py

from train import *
from yolo import *

#Preprocessing data parameters
input_shape = (416,416)
annotation_path_train = '2007_test.txt'
annotation_path_test = '2007_test.txt'
log_dir = 'logs/000/'
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

val_split = 0.1
training_lines = None
with open(annotation_path_train) as f:
  training_lines = f.readlines()

np.random.seed(10101)
np.random.shuffle(training_lines)
np.random.seed(None)
num_val = int(len(training_lines)*val_split)
num_train = len(training_lines) - num_val

batch_size = 16

#Pre-trained weights
path_teacher = "/content/DD2424-Project/keras-yolo3/model_data/yolov3_weights.h5"
path_student = "/content/DD2424-Project/keras-yolo3/model_data/yolov3-tiny_weights.h5"

#Fine-tuned_weights
path_teacher_fine_tuned = "/content/DD2424-Project/keras-yolo3/fine_tune_yolov3.h5"
path_student_fint_tuned = "/content/DD2424-Project/keras-yolo3/fine_tune_yolov3-tiny.h5"

#TODO FIX A STUDENT MODEL
anchors_tiny_path = 'model_data/tiny_yolo_anchors.txt'
anchors_tiny = get_anchors(anchors_tiny_path)

def create_pre_trained_model(input_shape, anchors, num_classes, weights_path):
  #TODO INTRODUCE CONVERTERT
  #based on repo keras-yolo3 train.py  .
  #This allows to create models from the previously created .h5 file and train them.

  model = create_model(input_shape, anchors, num_classes, weights_path)
  return model

def create_pre_trained_tiny_model(input_shape, anchors, num_classes, weights_path):
  #TODO INTRODUCE CONVERTER
  #based on repo keras-yolo3 train.py  .
  #This allows to create models from the previously created .h5 file and train them.

  model = create_tiny_model(input_shape, anchors, num_classes, weights_path= weights_path)
  return model

def create_fine_tuned_model(weight_path, anchors_path, classes_path):
# Must change in generate method:
# load_model add arg: by_name = True

  defaults = {
        "model_path": weight_path,
        "anchors_path": anchors_path,
        "classes_path": classes_path,
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }
  model = YOLO(**defaults)
  return model

def train_model(model, training_lines, num_train, batch_size, input_shape, anchors, num_classes):
  #based on train.py in repo
  lines = training_lines

  logging = TensorBoard(log_dir=log_dir)
  checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
      monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


  for i in range(len(model.layers)):
    model.layers[i].trainable = True

  model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change


  print('Unfreeze all of the layers.')

  model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    steps_per_epoch=max(1, num_train//batch_size),
    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    validation_steps=max(1, num_val//batch_size),
    epochs=100,
    initial_epoch=50,
    callbacks=[logging, checkpoint, reduce_lr, early_stopping])

  return model

#You can only train on model per time, because keras shares variables
#this causes the training to crasch if both models are created at the same session

#model = create_pre_trained_model(input_shape, anchors_tiny, num_classes, path_teacher)
#tiny_model = create_pre_trained_tiny_model(input_shape, anchors_tiny, num_classes, path_student)

#tiny_model.save_weights(path_student, overwrite = True)

# Train yolov3 model
#train_model = (training_lines, num_train, batch_size, input_shape, anchors, num_classes)
#Train tiny model
#train_model(tiny_model, training_lines, num_train, batch_size, input_shape, anchors_tiny, num_classes)

#teacher_model = create_fine_tuned_model(path_teacher_fine_tuned, anchors_path, classes_path)

#generator = test_generator(annotation_path_test, batch_size, input_shape, anchors, num_classes)

# These classes and methods are based on https://keras.io/examples/vision/knowledge_distillation/

class Distiller(tf.keras.Model):
  def __init__(self, student, teacher):
    super(Distiller, self).__init__()
    self.teacher = teacher
    self.student = student

  def compile(self,
              optimizer,
              metrices,
              student_loss,
              distillation_loss,
              alpha=0.1,
              temperature=3,
              ):

    super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
    self.student_loss = student_loss
    self.distillation_loss = distillation_loss
    self.alpha = alpha
    self.temperature = temperature



  # def train_step(self, data):
  #
  #   x,y = data
  #
  #   teacher_predictions = self.teacher.y_prediction(x)
  #
  #   with tf.GradientTape() as tape:
  #           # Forward pass of student
  #           student_predictions = self.student.predict(x)
  #
  #           student_loss = self.student_loss(y, student_predictions)
  #           distillation_loss = self.distillation_loss(
  #               tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
  #               tf.nn.softmax(student_predictions / self.temperature, axis=1),
  #           )
  #
  #           loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
  #
  #       # Compute gradients
  #       trainable_vars = self.student.trainable_variables
  #       gradients = tape.gradient(loss, trainable_vars)
  #
  #       # Update weights
  #       self.optimizer.apply_gradients(zip(gradients, trainable_vars))
  #
  #       # Update the metrics configured in `compile()`.
  #       self.compiled_metrics.update_state(y, student_predictions)
  #
  #       # Return a dict of performance
  #       results = {m.name: m.result() for m in self.metrics}
  #       results.update(
  #           {"student_loss": student_loss, "distillation_loss": distillation_loss}
  #       )
  #       return results

  # def eval(self, data):
  #
  #   x,y = data
  #
  #   y_predictions = self.student(x, training=False)
  #
  #   student_loss = self.student_loss(y, y_prediction)
  #
  #   self.compiled_metrics.update_state(y, y_prediction)
  #
  #   results = {m.name: me.result() for m in self.metrics}
  #   results.update({"student_loss": student_loss})
  #   return results


if __name__ == "__main__":
    print("Running Main Program")

# teacher = create_fine_tuned_model(path_teacher_fine_tuned, anchors_path, classes_path)
#
# student = create_fine_tuned_model(path_student, anchors_tiny_path, classes_path)
#
# knowlegde_distiller = Distiller(teacher, student)
# knowlegde_distiller.comile(optimizer=keras.optimizers.Adam(),
# metrics=[keras.metrics.SparseCategoricalAccuracy()],
# student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# distillation_loss_fn=keras.losses.KLDivergence(),
# alpha=0.1,
# temperature=10,
#
# knowlegde_distiller.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
# knowlegde_distiller.fit(x_train, y_train, epochs=3)
