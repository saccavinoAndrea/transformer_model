# src/immobiliare/cli/run_scraping.py
from pathlib import Path
import asyncio

from immobiliare.config_loader import ConfigLoader
from immobiliare.scraping.http_fetcher import HttpFetcher

from immobiliare.utils.logging.logger_factory import LoggerFactory


async def main():
    # Carica config dal package data
    config = ConfigLoader.load()

    # Configura i livelli dei logger
    LoggerFactory.configure(config.loggers)

    logger = LoggerFactory.get_logger("http_fetcher")
    logger.log_info("Inizio download HTML")

    from_page = config.from_page

    fetcher = HttpFetcher(
                logger=logger,
                root_url="https://www.immobiliare.it",
                base_url="https://www.immobiliare.it/vendita-case/firenze/?pag=",
                from_page=from_page
            )

    await fetcher.fetch(
        n_pages=config.n_pages,
        out_dir=Path(config.raw_html_dir)
    )

if __name__ == '__main__':
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
