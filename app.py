from flask import Flask, render_template, request
from werkzeug.utils  import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import shutil
    
model = tf.keras.models.load_model("model")
model.summary()
app = Flask(__name__,template_folder='Web')
app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = "uploaded/image"


@app.route('/')
def upload_f():
    return render_template('upload.html')


def finds():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    #vals = ['Covid Patient', 'Normal Person']  # change this according to what you've trained your model to do
    test_dir = 'uploaded'

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        shuffle=False,
        class_mode='binary',
        batch_size=1)

    pred = model.predict(test_generator)
    print(pred)
    
    if pred>0.75:
        return "Covid Patient"
    else:
        return 'Normal Person'
    



@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val = finds()
        shutil.rmtree('uploaded/image/')
        os.mkdir("uploaded/image/")
        return render_template("pred.html", ss=val)

 
#

if __name__ == '__main__':
    app.run()

    
