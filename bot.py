import telebot
token = '6935918318:AAGXB5Mdfcl9jHGRHAI66hO61foGhv8FQyA'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Загрузка...')
tokenizer = AutoTokenizer.from_pretrained('mc')
model = AutoModelForCausalLM.from_pretrained('mc').cpu()
tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})
print('Загрузка завершена')

bot = telebot.TeleBot(token)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message):
	prompt = "- {}\n-".format(message.text) # Название функции не играет никакой роли
	encoded_prompt = tokenizer.encode(prompt, return_tensors="pt").cpu()

	out = model.generate(encoded_prompt, max_length=200, do_sample=True, top_k=35, top_p=0.95, temperature=0.8,
							num_return_sequences=1, eos_token_id=2, pad_token_id=0)
	for i, tokens in enumerate(out.cuda().tolist(), start=1):
		tokens = tokens[encoded_prompt.shape[1]:]
		text = tokenizer.decode(tokens)
	#reply = text[:text.index('\n')]
	#print('[{}] - {}'.format(i, reply))
		bot.send_message(message.chat.id, text)


if __name__ == '__main__':
	bot.infinity_polling()



