from core.bot import ArbBot, ArbBotConfig


def main():
    config = ArbBotConfig()
    bot = ArbBot(config)
    bot.run()


if __name__ == "__main__":
    main()
