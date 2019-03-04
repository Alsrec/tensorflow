from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageFilter
import tensorflow as tf


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
from predict import Predict
from model import Network

#from scipy import misc

CKPT_DIR = 'model/'
log = logging.getLogger()

log.setLevel('INFO')

handler = logging.StreamHandler()

handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

log.addHandler(handler)

#from cassandra.cluster import Cluster

#from cassandra import ConsistencyLevel



KEYSPACE = "bigdata"
session = 0

'''def createKeySpace():
   global session'''
cluster = Cluster(contact_points=['test-cassandra'],port=9042)

session = cluster.connect('bigdata')


'''   log.info("Creating keyspace...")

   try:

       session.execute("""

           CREATE KEYSPACE bigdata

           WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }

           """ % KEYSPACE)


       log.info("setting keyspace...")

       session.set_keyspace(KEYSPACE)


       log.info("creating table...")

       session.execute("""

           CREATE TABLE mnist (


                   imgname text,
                   result text,

                   PRIMARY KEY (imgname)

           )

           """)

   except Exception as e:

       log.error("Unable to create keyspace")

       log.error(e)


   createKeySpace();'''


app = Flask(__name__)




result = -1
imgname=""
idnum=1



@app.route('/upload', methods=['POST'])
def upload():
    global  result
    global imgname
    global idnum
    f = request.files['file']
    img = Image.open(f.filename)
    app = Predict()
    result = app.predict(img)
    imgname = str(secure_filename(f.filename))
    # store to cassandra
    global session
    session.execute("""INSERT INTO mnist (imgname, result)
    VALUES ('""" + imgname + """','""" + str(result) + """')""")

    '''session.execute("""INSERT INTO mnist (id, imgname, result) VALUES (1,imgname,result)""")'''
    '''session.execute(
            """
            INSERT INTO demandtable (id, imgname, result)
            VALUES (%s, %s)
            """,
            (1,imgname,result)
            )'''
    return 
    ''' 
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
