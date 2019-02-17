from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import logging
from io import BytesIO
import datetime
from flask import Flask,request
import numpy as np
from cassandra.cluster import Cluster
from werkzeug.utils import secure_filename
from cassandra.query import SimpleStatement
import numpy as np
from PIL import Image

from model import Network

from scipy import misc

CKPT_DIR = 'model/'
log = logging.getLogger()

log.setLevel('INFO')

handler = logging.StreamHandler()

handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

log.addHandler(handler)

#from cassandra.cluster import Cluster

#from cassandra import ConsistencyLevel



KEYSPACE = "mykeyspace"
session = 0

def createKeySpace():
   global session
   cluster = Cluster(contact_points=['127.0.0.1'],port=9042)

   session = cluster.connect()


   log.info("Creating keyspace...")

   try:

       session.execute("""

           CREATE KEYSPACE %s

           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }

           """ % KEYSPACE)


       log.info("setting keyspace...")

       session.set_keyspace(KEYSPACE)


       log.info("creating table...")

       session.execute("""

           CREATE TABLE mytable (

               mykey text,

               col1 text,

               col2 text,

               PRIMARY KEY (mykey, col1)

           )

           """)

   except Exception as e:

       log.error("Unable to create keyspace")

       log.error(e)


   createKeySpace();


app = Flask(__name__)

class Predict:
    def __init__(self):
        self.net = Network()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

       
        self.restore()

    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        #else:
            #raise FileNotFoundError("no")

    def predict(self, file):
        
        #img = Image.open(image_path).convert('L')
        #img=img.convert('L')
        im = misc.imread(file)
        #file = im.reshape((1,784))
        flatten_img = np.reshape(file, 784)
        x = np.array([1 - flatten_img])
        y = self.sess.run(self.net.y, feed_dict={self.net.x: x})

        
       
        #print(image_path)
        print('        -> Predict digit', np.argmax(y[0]))
        x=str(np.argmax(y[0]))
        return x



result = -1
imgname=""
idnum=1



@app.route('/upload', methods=['POST'])
def upload():
    global  result
    global imgname
    global idnum
    f = request.files['file']
    img = f.read()
    app = Predict()
    result = app.predict(file)
    imgname = str(secure_filename(f.filename))
    # store to cassandra
    global session
    session.execute("""INSERT INTO record (id, imgname, img, time, result)
        VALUES (""" + str(idnum) + """,'""" + imgname + """','""" + img + """''""" + str(
        datetime.datetime.now()) + """','""" + str(result) + """')""")
    return ''' 
    <!doctype html> 
    <html> 
    <body> '''+str(result)+'''
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'> 
    <input type='submit' value='Upload'> 
    </form> 
    '''


@app.route('/')
def index():
    return ''' 
    <!doctype html> 
    <html> 
    <body> 
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'> 
    <input type='submit' value='Upload'> 
    </form> 
    '''

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
