from flask import render_template,request,Flask
import joblib
import os

app = Flask(__name__)

model_path = os.path.join('model', 'model.pkl')
vectorizer_path = os.path.join('model', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/', methods = ['GET', 'POST'])
def home():
    prediction = ""
    if request.method == 'POST':
        user_input = request.form['message']
        vect_input = vectorizer.transform([user_input])
        result = model.predict(vect_input)[0]
        prediction = "Spam" if result.lower() == "spam" else "Ham"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

