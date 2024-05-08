import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from questions import answer_question

load_dotenv()  # take environment variables from .env.

# Get the directory of the current Python script
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.abspath(os.path.join(current_file_directory, os.pardir))

tg_bot_token = os.getenv("TG_BOT_TOKEN")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv(parent_directory + '/lsu_pilot/processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
      answer = answer_question(df, question=update.message.text, debug=True)
      await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}]

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(
    chat_id=update.effective_chat.id,
    text="I'm a bot, please talk to me!"
  )
  
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
  messages.append({"role": "user", "content": update.message.text})
  completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
  )

  completion_answer = completion.choices[0].message
  messages.append(completion_answer)

  await context.bot.send_message(
    chat_id=update.effective_chat.id,
    text=completion_answer.content
  )

if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  chat_handler = CommandHandler('chat', chat)
  mozilla_handler = CommandHandler('mozilla', mozilla)
  
  application.add_handler(start_handler)
  application.add_handler(chat_handler)
  application.add_handler(mozilla_handler)

  application.run_polling()