import numpy as np
from keras.preprocessing import image
img = image.load_img("9999_right.jpeg",target_size=(224,224))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
print(output)