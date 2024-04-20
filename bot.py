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

# Command for getting the price of a cryptocurrency
@bot.command()
async def price(ctx, crypto: str):
    id = crypto.lower()  # Convert input to lowercase for consistency
    api_url = f'https://api.coincap.io/v2/assets/{id}'
    response = requests.get(api_url).json()

    # Check if the response contains valid data
    if 'data' in response:
        price = response['data']['priceUsd']
        await ctx.send(f'The current price of {id} is ${price}')
    else:
        await ctx.send(f"Couldn't find information for {id}.")

# Run the bot with the specified token
bot.run(TOKEN)
