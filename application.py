import os
import randomforestparallelthreading
#import cv2

from flask import Flask, request, render_template, send_from_directory


app = Flask(__name__)
length = 0

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    global length
    global destination
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = 'input.png'
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        #length = roadforesttest.main(destination)
        #resize()
        # return send_from_directory("images", filename, as_attachment=True)
        #return render_template("complete.html", image_name=filename)
        return "uploading completed"

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/imgprocess',methods=["GET"])
def imgprocess():
    global length
    filename = 'input.png'
    length = randomforestparallelthreading.main(destination)
    #resize()
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename)

@app.route('/gallery')
def get_gallery():
    global length
    image_names = ['input.png', 'out2.png']
    print(image_names)
    print(length)
    return render_template("gallery.html", image_names=image_names, l=length)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80,debug = True)
