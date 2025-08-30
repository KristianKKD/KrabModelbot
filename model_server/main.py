from flask import Flask, request, jsonify
from AIModel import AIModel

app = Flask(__name__)
model = AIModel()
model_busy = False

@app.route('/generate', methods=['POST'])
def generate():
    global model_busy
    
    if model_busy:
        return jsonify({'response': 'Model is busy, please try again.'}), 429

    data = request.get_json()
    prompt = data.get('prompt', '')
    context_list = data.get('context', [])
    context_str = ''.join(context_list)

    model_busy = True
    try:
        response = model.generate_response(context_str, prompt)
    finally:
        model_busy = False

    return jsonify({'response': response})

@app.route('/restart', methods=['POST'])
def restart():
    global model

    try:
        #cleanup
        if model:
            model.__del__()

        #create new
        model = AIModel()
        model.load_model()
        return jsonify({'status': 'Model restarted successfully'})
    except Exception as e:
        return jsonify({'status': f'Failed to restart: {e}'}), 500

if __name__ == '__main__':
    model.load_model()
    app.run(host='0.0.0.0', port=5000)