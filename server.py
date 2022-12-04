from generate import Generator
from flask import Flask, request
from flask_cors import CORS

app = Flask("app")
CORS(app)

generator = Generator()

@app.post("/api/generate")
def generate_lyric():
	content = request.json
	prompt = content["prompt"]
	artist = content["artist"]
	return {"artist": artist, "options": generator.generate(artist, prompt), "prompt": prompt}


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=4000, debug=True)
