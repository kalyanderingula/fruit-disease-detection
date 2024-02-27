import os
from flask import Flask, request, render_template, send_from_directory,session
import mysql.connector


app = Flask(__name__)
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['Apple Blotch','Aspergillus Fruit Rot','Bacterial Blight','Black Rot','Downy Mildew','Gray Mold','Powdery Mildew','Apple scab','RottenApples']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/registration')
def registration():
    return render_template("ureg.html",msg='Successfully Registered!!')

@app.route('/userlog',methods=['POST', 'GET'])
def userlog():
    global name, password
    global user

    if request.method == "POST":

        username = request.form['email']
        password1 = request.form['pass']
        print('p')
        mydb = mysql.connector.connect(host="localhost",user="root",passwd="",port=3307,database="fruit")
        cursor = mydb.cursor()
        sql = "select * from ureg where email='%s' and pass='%s'" % (username, password1)
        print('q')
        x = cursor.execute(sql)
        print(x)
        results = cursor.fetchall()
        print(results)
        if len(results) > 0:
            print('r')
            #session['user'] = username
            #session['id'] = results[0][0]
            #print(id)
            #print(session['id'])
            return render_template('userhome.html', msg="Login Success")
        else:
            return render_template('user.html', msg="Login Failure!!!")

    return render_template('user.html')


@app.route('/uregback',methods=['POST','GET'])
def uregback():
    if request.method=='POST':
        name=request.form['name']
        email=request.form['email']
        pwd=request.form['pass']
        ph=request.form['ph']
        gender=request.form['gender']

        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            port=3307,
            database="fruit"
        )
        mycursor = mydb.cursor()

        sql = "INSERT INTO ureg (name,email,pass,ph,gender) VALUES (%s, %s,%s,%s,%s)"
        val = (name,email,pwd,ph,gender)
        mycursor.execute(sql, val)
        mydb.commit()
    return render_template('user.html')
    print("Successfully Registered")

@app.route('/upload1')
def upload1():
    return render_template("upload.html",msg='Successfully logined!!')

@app.route('/userhome')
def userhome():
    return render_template("userhome.html",msg='Successfully logined!!')





@app.route("/upload", methods=["POST"])
def upload():
    # print('a')
    # target = os.path.join(APP_ROOT, 'images/')
    # # target = os.path.join(APP_ROOT, 'static/')
    # print(target)
    # if not os.path.isdir(target):
    #         os.mkdir(target)
    # else:
    #     print("Couldn't create upload directory: {}".format(target))
    # print(request.files.getlist("file"))
    # for upload in request.files.getlist("file"):
    #     print(upload)
    #     print("{} is the file name".format(upload.filename))
    #     filename = upload.filename
    #     destination = "/".join([target, filename])
    #     print ("Accept incoming file:", filename)
    #     print ("Save it to:", destination)
    #     upload.save(destination)
    if request.method=="POST":
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('images/', fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)

        #import tensorflow as tf
        import numpy as np
        from tensorflow.keras.preprocessing import image

        from tensorflow.keras.models import load_model
        new_model = load_model('alg/ANNModel.h5')
        new_model.summary()
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image=test_image/255
        test_image = np.expand_dims(test_image, axis = 0)
        result = new_model.predict(test_image)
       # prediction=classes[np.argmax(result)]

        prediction=classes[np.argmax(result)]

    #return send_from_directory("images", filename, as_attachment=True)
    return render_template("template.html",image_name=fn, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)

