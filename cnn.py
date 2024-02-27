
from keras.models import Sequential, load_model
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Flatten())


classifier.add(Dense(units=32,activation = 'relu'))

classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=9,activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('D:\\fruit disease detection using color,texture and ANN\\dataset\\train',
                                                target_size=(64,64),
                                                batch_size=12,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('D:\\fruit disease detection using color,texture and ANN\\dataset\\test',
                                            target_size=(64,64),
                                            batch_size=12,
                                            class_mode='categorical')

history=classifier.fit_generator(training_set,
                         samples_per_epoch=1212,
                         nb_epoch = 35,
                         validation_data = test_set,
                         nb_val_samples = 300)


from matplotlib import pyplot as plt
plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='training accuracy',color='green')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig("D:\\fruit disease detection using color,texture and ANN\\plot1.png")
plt.show()

classifier.save("D:\\fruit disease detection using color,texture and ANN\\model.h5")


mm=load_model(r'D:\fruit disease detection using color,texture and ANN\model.h5')

'''from PIL import Image
import cv2
f1=cv2.imread(r'D:\fruit disease detection using color,texture and ANN\dataset\train\Apple Blotch\1.jpg')
mm.predict(ti)

ti = image.load_img(r'D:\fruit disease detection using color,texture and ANN\dataset\train\Apple Blotch\1.jpg',target_size=(64, 64))
ti = image.img_to_array(ti)
ti = np.expand_dims(ti, axis=0)
result = new_model.predict(test_image)

d=training_set.class_indices
list(d.keys())[np.argmax(mm.predict(ti))]
classes[np.argmax(mm.predict(ti))]'''