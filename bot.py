import os
import discord
from dotenv import load_dotenv
import requests
from discord.ext import commands
from discord.ext.commands import MissingRequiredArgument

from keras.models import load_model

import openai 

#Set the OpenAI API key
openai.api_key = "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"

# Set the discord token
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
TOKEN = DISCORD_TOKEN

# Set up Discord intents
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

# Create a bot instance with the specified command prefix and intents
bot = commands.Bot(command_prefix='!', intents=intents)

# Event handler for when the bot is ready
@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

#CRYPTOCURRENCY COMMANDS

# Command for getting the price of a cryptocurrency
@bot.command(help="Get the current price of a cryptocurrency")
async def price(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            price = data[0]['priceUsd']
            await ctx.send(f'The current price of {symbol.upper()} is ${price}')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")



# Command for getting the market cap of a cryptocurrency
@bot.command(help="Get the market cap of a cryptocurrency")
async def marketcap(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            marketcap = data[0]['marketCapUsd']
            await ctx.send(f'The current market cap of {symbol.upper()} is ${marketcap}')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")


# Command for getting the 24h trading volume of a cryptocurrency
@bot.command(help="Get the 24h trading volume of a cryptocurrency")
async def tradingvolume(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            tradingvolume = data[0]['volumeUsd24Hr']
            await ctx.send(f'The trading volume in the last 24 hours of {symbol.upper()} is ${tradingvolume}')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")


# Command for getting the availabe supply for trading of a cryptocurrency
@bot.command(help="Get the available supply for trading of a cryptocurrency")
async def supply(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            supply = data[0]['supply']
            await ctx.send(f'The available supply for trading of {symbol.upper()} is ${supply}')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")


# Command for getting the max supply for trading of a cryptocurrency
@bot.command(help="Get the max supply for trading of a cryptocurrency")
async def maxsupply(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            maxsupply = data[0]['maxSupply']
            await ctx.send(f'The available supply for trading of {symbol.upper()} is ${maxsupply}')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")


# Command for getting the 24h change percentage of a cryptocurrency
@bot.command(help="Get the 24h change percentage of a cryptocurrency")
async def change(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            change = data[0]['changePercent24Hr']
            await ctx.send(f'The change in the last 24 hours of {symbol.upper()} is {float(change):.2f}%')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")


# Command for getting the 24h VWAP of a cryptocurrency
@bot.command(help="Get the 24h VWAP of a cryptocurrency")
async def vwap(ctx, symbol: str):
    api_url = f'https://api.coincap.io/v2/assets?search={symbol}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        data = response['data']
        if data:
            vwap = data[0]['vwap24Hr']
            await ctx.send(f'The VWAP of {symbol.upper()} is ${vwap}')
        else:
            await ctx.send(f"Couldn't find information for {symbol.upper()}.")
    else:
        await ctx.send(f"Couldn't find information for {symbol.upper()}.")



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

#Prediction 

import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from dotenv import load_dotenv

prediction_model = load_model('prediction_model.h5')

# Load the scaler
scaler = MinMaxScaler(feature_range=(0,1))

# Function to preprocess input data
def preprocess_data(data):
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    return scaled_data.reshape(1, -1, 1)

@bot.command(help="Predicts the price of Bitcoin for the next day")
async def predict(ctx):
    try:
        # Load your dataset
        current_date = datetime.datetime.now().date()
        start_date = current_date - datetime.timedelta(days=15)
        btc_data = yf.download('BTC-USD', start=start_date, end=current_date)

        # Extract the necessary data for prediction
        closedf = btc_data[['Close']].values

        # Preprocess data
        data = closedf 
        preprocessed_data = preprocess_data(data)

        # Predict using the model
        prediction = prediction_model.predict(preprocessed_data)
        predicted_price = scaler.inverse_transform(prediction.reshape(1, -1))

        # Get the date for the next day
        next_date = current_date + datetime.timedelta(days=1)

        # Send the prediction and date to Discord
        await ctx.send(f"Predicted price for BTC tomorrow: ${predicted_price[0][0]}")
    except Exception as e:
        await ctx.send(f"Error predicting price: {str(e)}")



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


#NEWS COMMANDS

# Command for getting news related on a specific topic - top 5 articles
@bot.command(help="Get the top 5 news articles related to a specific topic")
async def news(ctx, *, query: str):
    url = f'https://newsapi.org/v2/everything?q="{query}"&language=en&sortBy=publishedAt&apiKey=05c15891d7fc45dabaa105cb4432273b'
    response = requests.get(url)
    news_data = response.json()

    # Check if the response contains valid data
    if 'articles' in news_data and news_data['articles']:
        articles = news_data['articles'][:5]  # Get only the top 5 articles
        valid_articles = [article for article in articles if article['title'] != '[Removed]']
        for article in valid_articles:
            await ctx.send(f"**{article['title']}**\n{article['url']}")
            response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": "You are an expert in analyzing sentiment. You respond with either Positive, Negative, or Neutral."},
                {"role": "user", "content": article['content']}
            ])
            await ctx.send(f"Sentiment: {response.choices[0].message.content}")
    else:
        await ctx.send(f"No news found for {query}.")


# Command for getting news headlines from BBC News - top 5 headlines
@bot.command(help="Get the top 5 news headlines from BBC News")
async def topheadlines(ctx):
    url = f'https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey=05c15891d7fc45dabaa105cb4432273b'
    response = requests.get(url)
    news_data = response.json()

       # Check if the response contains valid data
    if 'articles' in news_data and news_data['articles']:
        articles = news_data['articles'][:5]  # Get only the top 5 articles
        valid_articles = [article for article in articles if article['title'] != '[Removed]']
        for article in valid_articles:
            await ctx.send(f"**{article['title']}**\n{article['url']}")
    else:
        await ctx.send(f"No headlines found.")

# Command for analysing crypto market sentiment - uses top 5 news articles related to 'crypto market today'
@bot.command(help="Analyze crypto market sentiment")
async def marketsentiment(ctx):
    query = "crypto market today"  # Change the query to your desired topic
    url = f'https://newsapi.org/v2/everything?q="{query}"&sortBy=publishedAt&apiKey=05c15891d7fc45dabaa105cb4432273b'
    response = requests.get(url)
    news_data = response.json()

    # Check if the response contains valid data
    if 'articles' in news_data and news_data['articles']:
        articles = news_data['articles'][:5]  # Get only the top 5 articles
        valid_articles = [article for article in articles if article['title'] != '[Removed]']
        for article in valid_articles:
            await ctx.send(f"**{article['title']}**\n{article['url']}")
            response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": "You are an expert in analyzing market sentiment. Analyse the market sentiment."},
                {"role": "user", "content": article['content']}
            ])
            await ctx.send(f"Sentiment: {response.choices[0].message.content}")
    else:
        await ctx.send(f"No news found for {query}.")


# Run the bot with the specified token
bot.run(TOKEN)