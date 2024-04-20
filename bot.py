import os
import discord
from dotenv import load_dotenv
import requests
from discord.ext import commands
from discord.ext.commands import MissingRequiredArgument

# Set the discord token
DISCORD_TOKEN = 'MTIyODc3MDE1Mjg3OTc1NTQwNQ.Guszxa.bzgLo-Yn9MbgrrnghueiJI4-q3XaH-uKqu7Q6E'
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
            await ctx.send(f'The change in the last 24 hours of {symbol.upper()} is {change}')
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


#NEWS COMMANDS

# Command for getting news related on a specific topic - top 5 articles
@bot.command(help="Get the top 5 news articles related to a specific topic")
async def news(ctx, *, query: str):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey=05c15891d7fc45dabaa105cb4432273b'
    response = requests.get(url)
    news_data = response.json()

    # Check if the response contains valid data
    if 'articles' in news_data and news_data['articles']:
        articles = news_data['articles'][:5]  # Get only the top 5 articles
        for article in articles:
            await ctx.send(f"**{article['title']}**\n{article['url']}")
    else:
        await ctx.send(f"No news found for {query}.")


# Command for getting news headlines from a specific country - top 5 articles
@bot.command(help="Get the top 5 news headlines from a specific country")
async def topheadlines(ctx, query: str):
    url = f'https://newsapi.org/v2/top-headlines?country={query}&apiKey=05c15891d7fc45dabaa105cb4432273b'
    response = requests.get(url)
    news_data = response.json()

    # Check if the response contains valid data
    if 'articles' in news_data and news_data['articles']:
        articles = news_data['articles'][:5]  # Get only the top 5 articles
        for article in articles:
            await ctx.send(f"**{article['title']}**\n{article['url']}")
    else:
        await ctx.send("No headlines found.")



################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


# Run the bot with the specified token
bot.run(TOKEN)