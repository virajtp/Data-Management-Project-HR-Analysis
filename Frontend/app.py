
import pickle
from flask import Flask, render_template, request,redirect #flask app
import numpy as np

#creating the flask app
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def water():
    return render_template('input.html')

# final page route
@app.route('/output', methods=['GET', 'POST'])
def getOutput():
    data_list = []
    # Read from the Form
    content = request.form
    gender = float(content["gender"])
    city_development_index = float(content["city_development_index"])
    city= int(content["city"])
    relevent_experience = float(content["relevent_experience"])
    enrolled_university = float(content["enrolled_university"])
    education_level = float(content["education_level"])
    major_discipline = float(content["major_discipline"])
    company_type = float(content["company_type"])
    company_size = float(content["company_size"])
    experience = float(content["experience"])
    last_new_job = float(content["last_new_job"])
    training_hours = float(content["training_hours"])

    if experience >= 21 : 
        experience = 21
    if experience <= 0 : 
        experience = 0

    input_list = []
    input_list.extend([gender, relevent_experience, enrolled_university, education_level, major_discipline, experience, company_size, company_type, 
          last_new_job,  city_development_index, training_hours, city])

    output = model(input_list)

    data_list.append(output)

    return render_template('output.html', data=data_list)

def model(input_list):

    input_list = np.reshape(np.array(input_list), (-1, 12))
    filename = 'model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(input_list)

    if result == 0:
        outVal = "Not looking for job change"
    else:
        outVal = "Looking for a job change"

    return outVal

if __name__ == '__main__':
    app.run()