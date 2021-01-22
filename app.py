from flask import Flask, request, render_template, flash, redirect
import os
import uuid

from src.dyslexiaCheck import runInference as dyslexia
from src.personalityCheck import runInference as personality
# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    #return app.send_static_file('index.html')
    return render_template('index.html')

@app.route('/home')
def application():
    #return app.send_static_file('index.html')
    return render_template('home.html')

@app.route("/upload-image-dyslexia", methods=["GET", "POST"])
def upload_image_dyslexia():
    try:
        if request.method == "POST":
            if request.files:
                image = request.files["image"]
                imageDirectory = os.path.join(os.getcwd(),"uploads/dyslexia")
                uniqueFilename = str(uuid.uuid4())+"."+image.filename.split('.')[-1]
                imageDirectory = os.path.join(imageDirectory, uniqueFilename)
                image.save(imageDirectory)
                print("Image saved")
                #return redirect(request.url)
        response = dyslexia(imageDirectory)
        return render_template("dyslexia-report.html", pyArgs = response)
    except:
        return render_template("error.html")


@app.route("/upload-image-personality", methods=["GET", "POST"])
def upload_image_personality():
    try:
        if request.method == "POST":
            if request.files:
                image = request.files["image"]
                imageDirectory = os.path.join(os.getcwd(),"uploads/personality")
                uniqueFilename = str(uuid.uuid4())+"."+image.filename.split('.')[-1]
                imageDirectory = os.path.join(imageDirectory, uniqueFilename)
                image.save(imageDirectory)
                print("Image saved")
                #return redirect(request.url)
        response = personality(imageDirectory)
        return render_template("personality-report.html", pyArgs = response)
    except:
        return render_template("error.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='80')
#     app.run()
