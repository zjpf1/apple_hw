from flask import Flask, request, jsonify, render_template, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import json
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# a dummy secret key
app.secret_key = "randomvalue"
# create local database
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class Record(db.Model):
    """Create prediction history record table."""
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(50), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)

    def __init__(self, model, label, probability):
        self.model = model
        self.label = label
        self.probability = probability

class User(db.Model):
    """Create user table that has access to the history record."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def __init__(self, name, password_hash):
        self.name = name
        self.password_hash = password_hash

def set_password(password):
    """Convert the plain password to hashed password."""
    return generate_password_hash(password)

def check_password(password_hash, password):
    """Check whether the input password is correct by
    comparing to the hashed password stored in the database.
    """
    return check_password_hash(password_hash, password)

# A modified tf code, used to read local decode json file
def decode_predictions(preds, top=5):
    """Decodes the prediction of an ImageNet model.

    Arguments:
        preds: Numpy array encoding a batch of predictions.
        top: Integer, how many top-guesses to return. Defaults to 5.

    Returns:
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    Raises:
        ValueError: In case of invalid shape of the `pred` array
          (must be 2D).
    """

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    with open("./imagenet_class_index.json") as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def load_model():
    """Load MobileNetV2 model."""
    global model
    model = MobileNetV2(weights="./mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5")

def img_preprocessing(image, size=(224,224)):
    """Convert image to feed the model.
    Arguments:
        image: PIL format file
        size: tuple, indicates the resized image size
    """
    # convert image to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(size)
    image = img_to_array(image)
    # add 1 dim to fit model input size
    image = np.expand_dims(image, axis=0)
    image_input = imagenet_utils.preprocess_input(image, mode='tf')
    return image_input

def run_model(image, model):
    """Do the image prediction.
    Arguments:
        image: PIL format file
        model: Keras loaded model
    """
    model_type = "MobileNetV2"
    # Image pre-processing
    image = img_preprocessing(image)
    # Image classification
    predictions = model.predict(image)
    # Obtain classification results
    results = decode_predictions(predictions, top=1)
    for (imagenetID, label, prob) in results[0]:
        result = {"model": model_type, "imagenetID": imagenetID, "label": label, "probability": float(prob)}
    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST", "GET"])
def register():
    """User registration"""
    if request.method == "POST":
        name = request.form["name"]
        password = request.form["password"]
        # Error if name or password is not filled
        if not name:
            return "Missing name", 400
        if not password:
            return "Missing password", 400
        user = User.query.filter_by(name=name).first()
        # Only register when the username is unique
        if not user:
            password_hash = set_password(password)
            # Add to database
            db.session.add(User(name, password_hash))
            db.session.commit()
            # Auto redirect to login page
            return redirect(url_for("login"))
        else:
            return "User name has been registered!"
    else:
        return render_template("register.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    """User login"""
    if request.method == "POST":
        name = request.form["name"]
        password = request.form["password"]
        # Error if name or password is not filled
        if not name:
            return "Missing name", 400
        if not password:
            return "Missing password", 400
        user = User.query.filter_by(name=name).first()
        if not user:
            return "User Not Found!", 404
        # Redirect to history record page if username
        # and password are correct
        if check_password(user.password_hash, password):
            # Add login status to session for later check
            session["login"] = True
            return redirect(url_for("history"))
        else:
            return "wrong password!", 400
    else:
        return render_template("login.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Image prediction"""
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            # Load image via PIL
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # Create key to store result
            data["results"] = []
            # Run model
            result = run_model(image, model)
            # Add classification result to dict
            data["results"].append(result)
            # Add to database
            db.session.add(Record(result["model"], result["label"], result["probability"]))
            db.session.commit()
            # Update prediction status
            data["success"] = True
            # return dict as json
            return jsonify(data)
        else:
            return "Missing image", 400

@app.route("/history", methods=["GET"])
def history():
    """Show historical prediction record"""
    # User can only check the result when logged in
    if "login" in session:
        # Pop session element to ensure the need of log in everytime,
        # also possible to write a log out route separately if this is
        # not preferable
        session.pop("login", None)
        return render_template("history.html", values=Record.query.all())
    else:
        return "Invalid access: Login required!"

if __name__ == "__main__":
    print("Loading classification model and starting server...")
    db.create_all()
    load_model()
    app.run(debug=True)