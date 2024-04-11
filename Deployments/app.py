# Importing necessary Flask components and the predictive model as 'model'
from flask import Flask, request, render_template
import model

# Creating an instance of the Flask class.
app = Flask(__name__)


# Defining a route for the home page that accepts both GET and POST requests
@app.route("/", methods=["POST", "GET"])
def Home():
    # Initializing variables to store user inputs and the prediction outcome
    area = ""
    bedrooms = ""
    bathrooms = ""
    location = ""
    age = ""
    garage = ""
    price = ""

    # Processing form data sent via POST
    if request.method == "POST":
        # Retrieving data from the form
        area = request.form["area"]
        bedrooms = request.form["bedrooms"]
        bathrooms = request.form["bathrooms"]
        location = request.form["location"]
        age = request.form["age"]
        garage = request.form["garage"]

        # Using the model to predict the price based on the input data
        price = model.predict(area, bedrooms, bathrooms, location, age, garage)

    # Rendering an HTML template to display the form and, if available, the prediction
    return render_template(
        "index.html",
        outcome=price,
        area_input=area,
        bedrooms_input=bedrooms,
        bathrooms_input=bathrooms,
        location_input=location,
        age_input=age,
        garage_input=garage,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
