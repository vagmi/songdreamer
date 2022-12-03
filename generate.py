from aitextgen import aitextgen
from transformers import AutoTokenizer

ARTISTS = {
  "eminem": "./models/eminem-eleutherai--gpt-neo-125m",
  "metallica": "./models/metallica-eleutherai--gpt-neo-125m",
  "katy-perry": "./models/katy-perry-eleutherai--gpt-neo-125m"
}

class Generator:
	def __init__(self):
		self.models = {}

	def load_model(self, artist):
		model_folder = ARTISTS[artist]
		return aitextgen(model_folder=f"./{model_folder}", to_gpu=True)

	
	def model_for(self, artist):
		if self.models.get(artist) == None:
			self.models[artist] = self.load_model(artist)
		return self.models[artist]
			
	
	def generate(self, artist, prompt):
		model=self.model_for(artist)
		results = model.generate(n=3, prompt=prompt,
								 max_length=128, early_stopping=True,
								 temperature=1.5, no_repeat_ngram_size=2,
								 num_beams=5)
		return results

if __name__ == '__main__':
	prompt = "Working through the night\nnot giving up without a fight\nGoing off on a tangent\n"
	generator = Generator()
	print(generator.generate("eminem", prompt))
	print(generator.generate("metallica", prompt))
	print(generator.generate("katy-perry", prompt))