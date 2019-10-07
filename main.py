from flask import Flask, render_template, request, jsonify

import output
import load_model

import os

app = Flask(__name__)

model,source_ids,target_words,tokenizer_obj=load_model.load_model()

def create_response(mes):
	query=output.output(mes,source_ids,target_words,model,tokenizer_obj)
	return query

@app.route("/")
def chat():
	return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
	# メッセージを取得
	message = str(request.form['messageText'])

	while True:
		if message == "quit":
			exit()

		else:
			bot_response = create_response(message)
			# チャットボットの返答を返す
			return jsonify({'status':'OK','answer':bot_response})

if __name__ == "__main__":
	app.run(host='0.0.0.0', debug=True)