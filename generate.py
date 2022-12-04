from aitextgen import aitextgen
from transformers import AutoTokenizer
from typing import List

ARTISTS = {
  "eminem": "./models/eminem-eleutherai--gpt-neo-125m",
  "metallica": "./models/metallica-eleutherai--gpt-neo-125m",
  "katy-perry": "./models/katy-perry-eleutherai--gpt-neo-125m"
}
# ARTISTS = {
#   "eminem": "./models/eminem-eleutherai--gpt-neo-1.3b",
#   "metallica": "./models/metallica-eleutherai--gpt-neo-1.3b",
#   "katy-perry": "./models/katy-perry-eleutherai--gpt-neo-1.3b"
# }

MODEL="EleutherAI/gpt-neo-125M"
class Generator:
	def __init__(self):
		self.models = {}
		self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

	def load_model(self, artist: str) -> aitextgen:
		model_folder = ARTISTS[artist]
		return aitextgen(model_folder=f"./{model_folder}", to_gpu=True)

	
	def model_for(self, artist: str) -> aitextgen:
		if self.models.get(artist) == None:
			self.models[artist] = self.load_model(artist)
		return self.models[artist]
			
	
	def generate(self, artist: str, prompt: str) -> List[str]:
		model=self.model_for(artist)
		token_length = len(self.tokenizer.tokenize(prompt)) * 2
		token_length = token_length if token_length > 64 else 64
		results = model.generate(n=3, prompt=prompt,
								 min_length=32,
								 max_length=token_length, early_stopping=True,
								 temperature=1.5, no_repeat_ngram_size=2,
								 return_as_list=True,
								 num_beams=5)
		return results

if __name__ == '__main__':
	prompt = "Working through the night\nnot giving up without a fight\nGoing off on a tangent\n"
	prompt2 = """
	Oh, halo on fire
The midnight knows it well
Fast, is desire
Creates another hell
I fear to turn on the light
For the darkness won't go away
Fast, is desire
Turn out the light
Halo on fire
	"""
	generator = Generator()
	print(generator.generate("eminem", prompt))
	print(generator.generate("metallica", prompt))
	print(generator.generate("katy-perry", prompt))
	print(generator.generate("eminem", prompt2))
	print(generator.generate("metallica", prompt2))
	print(generator.generate("katy-perry", prompt2))