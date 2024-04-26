Cryptocurrency and News Discord Bot

This Discord bot provides various functionalities related to cryptocurrency information and news analysis. It utilizes the CoinCap API for cryptocurrency data and the News API for fetching news articles. Additionally, it employs OpenAI's GPT-3.5 model for sentiment analysis.
Installation

To use this bot, you need to follow these steps:

    Clone this repository to your local machine.
    Install the required dependencies listed in the requirements.txt file by running:

pip install -r requirements.txt

Create a .env file in the root directory of the project and define the following environment variables:

    DISCORD_TOKEN=<your_discord_token>
    GPT_KEY=<your_openai_gpt_key>

    Replace <your_discord_token> with your Discord bot token and <your_openai_gpt_key> with your OpenAI GPT API key.
    

Usage

Once you have set up the bot and installed the dependencies, you can run the bot using the following command:

python bot.py

Commands
Cryptocurrency Commands

    !price <symbol>: Get the current price of a cryptocurrency.
    !marketcap <symbol>: Get the market cap of a cryptocurrency.
    !tradingvolume <symbol>: Get the 24h trading volume of a cryptocurrency.
    !supply <symbol>: Get the available supply for trading of a cryptocurrency.
    !maxsupply <symbol>: Get the max supply for trading of a cryptocurrency.
    !change <symbol>: Get the 24h change percentage of a cryptocurrency.
    !vwap <symbol>: Get the 24h VWAP of a cryptocurrency.

Prediction Command

    !predict: Predicts the price of Bitcoin for the next day.

News Commands

    !news <query>: Get the top 5 news articles related to a specific topic.
    !topheadlines: Get the top 5 news headlines from BBC News.
    !marketsentiment: Analyze crypto market sentiment using the top 5 news articles related to 'crypto market today'.

Note

This bot relies on external APIs for fetching cryptocurrency data and news articles. Ensure that these APIs are accessible and functioning properly for the bot to work as expected.
