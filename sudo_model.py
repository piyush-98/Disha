from scipy.misc import imread
from scipy.misc import imresize




#image = cv2.imread(imagePath)
#im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.predict_classes(im, verbose=1)

