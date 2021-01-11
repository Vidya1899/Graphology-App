from flask import Flask, request, render_template, flash, redirect
import os
import uuid

from src.detector import runInference

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

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            imageDirectory = os.path.join(os.getcwd(),"uploads")
            uniqueFilename = str(uuid.uuid4())+"."+image.filename.split('.')[-1]
            imageDirectory = os.path.join(imageDirectory, uniqueFilename)
            image.save(imageDirectory)
            print("Image saved")
            #return redirect(request.url)
    response = runInference(uniqueFilename)
    return render_template("success.html", pyArgs = response)


if __name__ == "__main__":
    app.run()
