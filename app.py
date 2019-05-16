from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/app', methods=['POST'])
def process():

    if request.method =='POST':

        loaded_model = joblib.load('model.sav')
        pclass = request.form['pclass']
        age = request.form['age']
        sex = request.form['sex']
        travel_alone = request.form['travel_alone']
        data = {'Age': [age], 'TravelAlone': [0], 'Pclass_1': [0], 'Pclass_2': [0], 'Pclass_3': [0], 'Sex_female': [0], 'Sex_male': [0]}

        if travel_alone == 'alone':
            data['TravelAlone'] = 1

        if pclass == 1:
            data['Pclass_1'] = 1
        elif pclass == 2:
            data['Pclass_2'] = 1        
        else:
            data['Pclass_3'] = 1

        if sex == 'male' :
            data['Sex_male'] = 1
        else: 
            data['Sex_female'] = 1

        v = pd.DataFrame.from_dict(data)

        done = round(loaded_model.predict_proba(v)[0,1], 3)
        d = str(done)
        return jsonify({'name' : d})

    return jsonify({'name' : d})

if __name__ == '__main__':
	app.run()