## ML-DL-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Paper Link which explains it's working: [Refer](https://www.researchgate.net/publication/341873237_Stock_Prediction_using_Hybrid_ARIMA_and_GRU_Models)

### Prerequisites
You must have Scikit Learn, Pandas, Statsmodel,Tensorflow(version 1.14) and Keras(version 2.2.5) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. gru.h5 - Pretrained model of **GATED RECURRENT UNIT**.This file contains the whole structure and trained weights of GRU.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.

### Running the project
1. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

2. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
<p align = "center">
<img src = "/ims/im.png" width = 480>
</p>

Enter a valid day in the input boxe and hit Predict.

If everything goes well, you should  be able to see the predcited trend on the HTML page!
<p align = "center">
<img src = "/ims/im1.png" width = 480>
</p>

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
