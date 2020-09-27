# An introduction to the software API

This document provides the detailed information for each API within this web app.

## 0. Operation environment:

The code is written in Python 3.6.3 and the following packages are required:

- Flask == 1.1.2

- Flask-SQLAlchemy == 2.4.4

- Werkzurg == 1.0.1

- tensorflow == 2.3.0 (MS VC++ 2015-2019 redistributable is required)

- numpy == 1.18.5

- Pillow == 7.2.0

User can install these packages via '''pip install'''

## 1. Home page ('http://127.0.0.1:5000/'):

Home page contains a welcome sentence and 3 navigation buttons on the top: Home, Register and Login.

User can click "Register" or "Login" to redirect to the registration or login page.

## 2. Registration page ('http://127.0.0.1:5000/register'):

This page is used to register the account for model prediction history record table. 

The username for registration should be less than 50 letters.

Registration will fail if the username has been registered or the password or the username is empty.

In this case simply click "go back" on your web browser to restart registration.

## 3. Login page ('http://127.0.0.1:5000/login'):

This page is used to allow registered user to log in to the model prediction history record table.

Log in will fail if the username has not been registered or the password or the username is empty.

In this case user need to access 'http://127.0.0.1:5000/register' to do the registeration first or click "go back" on the web browser to fill up the username and password.

## 4. Predict request ('http://127.0.0.1:5000/predict'):

This request is used to receive image and return the classification model prediction result.

The prediction model used here is MobileNetV2, as it is small while gives acceptable detection accuracy.

As this request only accepts the POST method, it cannot be accessed via web browser.

In order to do the prediction, cURL is required and user can download it from 'https://curl.haxx.se/download.html'

The general format to do the prediction in windows cmd via cURL is:

'''
curl -X POST -F image=@your_image_location.png http://127.0.0.1:5000/predict
'''

Note the your_image_location.png indicates the user's local image address for prediction, e.g. '''./image_test/dog.png'''

If the prediction is successful, the user will receive the prediction result in the following format:

'''
{"results": [{"imagenetID": "n02108422", "label": "bull_mastiff", "model": "MobileNetV2", "probability": 0.9596219062805176}], "success": true}
'''

The prediction result will automatically stored in the model prediction history record table.

## 5. History page ('http://127.0.0.1:5000/history'):

This page shows all the historical model prediction results.

This page cannot be accessed directly, if do so the "Invalid access: Login required!" error will be shown on the page.

User has to log in to access this page via 'http://127.0.0.1:5000/login'
