from flask import Flask, request, jsonify
import numpy as np
from preprocess import preproces, model

app = Flask(__name__)


# Endpoint to provide mental health support suggestions based on user information
@app.route('/price_pred', methods=['POST'])
def get_support():
    try:
        # Extract user information from the incoming JSON request
        data = request.get_json()
        house_details = request.get_json()
        h_type = house_details['type']
        h_size = house_details['size']
        h_bedrooms = house_details['bedrooms']
        h_bathrooms = house_details['bathrooms']
        h_floor = house_details['floor']
        h_fur = house_details['fur']
        h_region = house_details['region']
        h_city = house_details['city']

        num_rooms = h_bathrooms+h_bedrooms

        h_type, h_fur, h_region, h_city = preproces(h_type, h_fur, h_region, h_city)
        output = np.array([h_type, h_size, h_bedrooms, h_bathrooms, h_floor, h_fur, h_region, h_city, num_rooms])
        price = model.predict(output.reshape(1, -1))



        # Return the mental health support suggestions in JSON format
        return jsonify(price[0])

    except Exception as e:
        # Handle exceptions and return an error message in JSON format
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)