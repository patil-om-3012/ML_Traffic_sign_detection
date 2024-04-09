import numpy as np
import os
from flask import *
from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

class_dict = {
    0:'Speed limit (20kmph)',
    1:'Speed limit (30kmph)',
    2:'Speed limit (50kmph)',
    3:'Speed limit (60kmph)',
    4:'Speed limit (70kmph)',
    5:'Speed limit (80kmph)',
    6:'End of speed limit (80kmph)',
    7:'Speed limit (100kmph)',
    8:'Speed limit (120kmph)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Vehicle over 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing vehicle > 3.5 tons'
}

def predict_img(img):
    cur_path = os.getcwd()
    model_path = os.path.join(cur_path,'training_set','model.h5')
    model = load_model(model_path)
    dataset = []
    image = Image.open(img)
    image = image.resize((30,30))
    dataset.append(np.array(image))
    X_test = np.array(dataset)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis = 1)
    return y_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload_img():
    if request.method == 'POST':
        inp = request.files['file']
        file_path = secure_filename(inp.filename)
        inp.save(file_path)
        prediction = predict_img(file_path)
        s = [str(i) for i in prediction]
        a = int("".join(s))
        prediction = 'The above sign intends to say ' + class_dict[a]
        os.remove(file_path)
        return prediction
    return None

if __name__ == '__main__':
    app.run(debug=True)