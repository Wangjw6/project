from flask import Flask, request, render_template
import tensorflow as tf
import model
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

app = Flask(__name__)

availableRoad = pd.DataFrame(pd.read_csv('/mnt/d/wjw/traffic_predict/dataset/available_road.csv',encoding='gb2312'))

def process(datapath,timestep=9,road=189):
    data = pd.DataFrame(pd.read_csv(datapath))
    X = np.array(data.iloc[:, 0:timestep*road])
    Y = np.array(data.iloc[:, timestep*road:])
    scaler = joblib.load('/mnt/d/wjw/model/trainDir/Scaler_CNN_45min_5min_road.save')
    rescaledX = scaler.transform(X)
    return X ,rescaledX, Y

def generateBatch(X,rescaledX,Y,batchsize=1,train=False,road=189,timestep=9):
    originalex_batches = []
    rescaledx_batches = []
    y_batches = []
    for i in range(batchsize):
        originalx_batch=[]
        rescaledx_batch = []
        y_batch = []
        if train:
            index = random.randint(0, X.shape[0]-1)
        else:
            index = i

        for k in range(int(len(X[index])/road)): # iterate according to timestep
            originalx_batch = np.append(originalx_batch, X[index][k*road:(k+1)*road])
            rescaledx_batch = np.append(rescaledx_batch, rescaledX[index][k*road:(k+1)*road]) # np.transpose(np.reshape(np.array(X[index][k*50:(k+1)*50]), (2,5,5)),(1,2,0))# 5 road * 5 road * 2 kind of direction

        y_batch = Y[index]
        originalex_batches.append(originalx_batch)
        rescaledx_batches.append(rescaledx_batch)
        y_batches.append(y_batch)

    return originalex_batches, rescaledx_batches, y_batches

@app.route('/', methods=['GET', 'POST'])
def index():

    # accept road log and lat to predict and return
    if request.method == "POST":
        a = request.form['lnglat'].split(',')
        b = request.form['roadname']

        predict_horizon = request.form.get('predict_horizon')

        print(a)
        lng = float(a[0])
        lat = float(a[1])
        dist = 9999
        index=-1

        for i in range(availableRoad.shape[0]):
            beginlat=availableRoad.iloc[i,5]
            beginlng=availableRoad.iloc[i,4]
            endlat=availableRoad.iloc[i,6]
            endlng=availableRoad.iloc[i,7]

            if min(abs(beginlat-lat)+abs(beginlng-lng), abs(endlat-lat)+abs(endlng-lng))<dist:
                dist = min(abs(beginlat-lat)+abs(beginlng-lng), abs(endlat-lat))
                index = i


        if int(predict_horizon)==5:
            print(predict_horizon)
            sess = tf.Session()
            m = model.CNN()
            m.build_CNN()
            saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.latest_checkpoint('/mnt/d/wjw/model//trainDir/logs_c_45min_5min_road/')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("## restore from the checkpoint {0}".format(checkpoint))

            X_test1_original, X_test1, Y_test1 = process(
                datapath='/mnt/d/wjw/traffic_predict/dataset/sample_45min_5min_test.csv')
            originalx_batch, x_batch, y_batch = generateBatch(batchsize=1, rescaledX=X_test1, X=X_test1_original, Y=Y_test1, train=False)

            predict=sess.run(m.predict,feed_dict={m.bottom:x_batch})
            print(predict[0][index])
            result = ''
            for i in range(9):
                # result+=str(x_batch[0][index+i*189])+','
                result += str(originalx_batch[0][index+i*189]) + ','
            result+=str(predict[0][index])

            print(result)
            return render_template('index.html', RESULT = result, ROAD=b, Predict_Horizon='5')

        if int(predict_horizon)==15:
            sess = tf.Session()
            m = model.CNN15()
            m.build_CNN()
            saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.latest_checkpoint('/mnt/d/wjw/model//trainDir/logs_c_45min_15min_road/')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("## restore from the checkpoint {0}".format(checkpoint))

            X_test1_original, X_test1, Y_test1 = process(
                datapath='/mnt/d/wjw/traffic_predict/dataset/sample_45min_15min_test.csv')
            originalx_batch, x_batch, y_batch = generateBatch(batchsize=1, rescaledX=X_test1, X=X_test1_original,
                                                              Y=Y_test1, train=False)

            predict = sess.run(m.predict, feed_dict={m.bottom: x_batch})
            print(predict[0][index])
            result = ''
            for i in range(9):
                # result+=str(x_batch[0][index+i*189])+','
                result += str(originalx_batch[0][index + i * 189]) + ','
            for j in range(3):
                if j < 2:
                    result += str(predict[0][index + j * 189]) + ','
                else:
                    result += str(predict[0][index + j * 189])

            print(result)
            return render_template('index.html', RESULT=result, ROAD=b, Predict_Horizon='15')

        if int(predict_horizon)==30:
            sess = tf.Session()
            m = model.CNN30()
            m.build_CNN()
            saver = tf.train.Saver(tf.global_variables())
            checkpoint = tf.train.latest_checkpoint('/mnt/d/wjw/model//trainDir/logs_c_45min_30min_road/')
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("## restore from the checkpoint {0}".format(checkpoint))

            X_test1_original, X_test1, Y_test1 = process(
                datapath='/mnt/d/wjw/traffic_predict/dataset/sample_45min_30min_test.csv')
            originalx_batch, x_batch, y_batch = generateBatch(batchsize=1, rescaledX=X_test1, X=X_test1_original,
                                                              Y=Y_test1, train=False)

            predict = sess.run(m.predict, feed_dict={m.bottom: x_batch})
            # print(predict[0][index])
            result = ''
            for i in range(9):
                # result+=str(x_batch[0][index+i*189])+','
                result += str(originalx_batch[0][index + i * 189]) + ','
            for j in range(6):
                if j<5:
                    result += str(predict[0][index + j * 189])+ ','
                else:
                    result += str(predict[0][index + j * 189])

            print(result)
            return render_template('index.html', RESULT=result, ROAD=b, Predict_Horizon='30')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
