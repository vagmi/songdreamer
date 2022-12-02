from datasets import load_dataset

from aitextgen.TokenDataset import TokenDataset
from transformers import AutoTokenizer

from aitextgen import aitextgen


MODEL="EleutherAI/gpt-neo-125M"
# MODEL = "EleutherAI/gpt-neo-1.3B"

# ARTIST_NAME = "eminem"
# ARTIST_NAME = "katy-perry"
ARTIST_NAME = "metallica"

class Trainer:
	def __init__(self, model, artist_name):
		self.model = model
		self.artist_name = artist_name
		self.ai = aitextgen(model=model, to_gpu=True)
		self.tokenizer = AutoTokenizer.from_pretrained(self.model)
		self.tokenizer.pad_token = self.tokenizer.eos_token
	
	def file_name(self):
		return f"lyric_texts/{ARTIST_NAME}.txt"

	def normalized_model_name(self):
		return self.model.replace("/", "--").lower()

	def model_dir(self):
		return f"models/{self.artist_name}-{self.normalized_model_name()}"

	def download_dataset(self):
		self.ds = load_dataset(f"huggingartists/{self.artist_name}")
		f=open(self.file_name(), 'w')
		content = "\n".join([f"{self.tokenizer.bos_token}{x}{self.tokenizer.eos_token}\n" for x in self.ds["train"]["text"]])
		f.write(content)
		f.close()

	def train(self):
		data = TokenDataset(self.file_name(), tokenizer=self.tokenizer, block_size=64)
		self.ai.train(data,
					  output_dir=self.model_dir(),
					  batch_size=1, 
					  num_steps=5000, 
					  save_every=1000, 
					  generate_every=500)


if __name__ == '__main__':
	artists = ['metallica', 'eminem', 'katy-perry']
	for artist in artists:
		trainer = Trainer(MODEL, artist)
		trainer.download_dataset()
		trainer.train()
		del trainer