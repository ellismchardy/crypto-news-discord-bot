import os
import discord
from dotenv import load_dotenv
import requests
from discord.ext import commands
from discord.ext.commands import MissingRequiredArgument


DISCORD_TOKEN = 'MTIyODc3MDE1Mjg3OTc1NTQwNQ.Guszxa.bzgLo-Yn9MbgrrnghueiJI4-q3XaH-uKqu7Q6E'
TOKEN = DISCORD_TOKEN

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def btc(ctx, id: str = 'bitcoin'):
    id = 'bitcoin'
    api_url = f'https://api.coincap.io/v2/assets/{id}'
    response = requests.get(api_url).json()
    price = response['data']['priceUsd']
    await ctx.send(f'The current price of {id} is ${price}')

@bot.command()
async def eth(ctx, id: str = 'bitcoin'):
    id = 'ethereum'
    api_url = f'https://api.coincap.io/v2/assets/{id}'
    response = requests.get(api_url).json()
    price = response['data']['priceUsd']
    await ctx.send(f'The current price of {id} is ${price}')

@bot.command()
async def crypto(ctx, id: str):
    id = ctx.command.name
    api_url = f'https://api.coincap.io/v2/assets/{id}'
    response = requests.get(api_url).json()
    price = response['data']['priceUsd']
    await ctx.send(f'The current price of {id} is ${price}')

   







bot.run(TOKEN)