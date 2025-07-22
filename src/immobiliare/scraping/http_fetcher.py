# src/immobiliare/scraping/http_fetcher.py
import socket
from typing import Optional
from pathlib import Path
import asyncio

from playwright.async_api import async_playwright, Playwright, TimeoutError as PlaywrightTimeoutError
from immobiliare.core_interfaces.fetcher.abstract_ifetcher import IFetcher
from immobiliare.core_interfaces.fetcher.ipage_fetcher import IPageFetcher
from immobiliare.core_interfaces.logger.ilogger import ILogger  # interfaccia
from immobiliare.utils.logging.decorators import alog_exec


class HttpFetcher(IFetcher, IPageFetcher):
    """
    Fetcher HTTP basato su Playwright che scarica pagine di annunci immobiliari.
    """

    def __init__(self,
                 logger: ILogger,
                 root_url: str = "https://www.immobiliare.it",
                 base_url: str = "https://www.immobiliare.it/vendita-case/roma/?pag=",
                 from_page: int = 1,
                 user_agent: Optional[str] = None,
                 viewport: Optional[dict] = None,
                 locale: str = "it-IT",
                 max_attempts: int = 3):

        self.logger = logger
        self.root_url = root_url
        self.base_url = base_url
        self.from_page = from_page
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
        self.viewport = viewport or {"width": 1280, "height": 800}
        self.locale = locale
        self.max_attempts = max_attempts


    @alog_exec(logger_name="fetcher", method_name="scrape_data")
    async def scrape_data(self, playwright: Playwright, n_pages: int, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            browser = await playwright.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(
                user_agent=self.user_agent,
                viewport=self.viewport,
                locale=self.locale,
                extra_http_headers={
                    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7"
                }
            )
            page = await context.new_page()

            # Blocca DataDome
            await page.route("**/*captcha-delivery.com/*", lambda route, _: route.abort())
            await page.route("**/*ct.captcha-delivery.com/*", lambda route, _: route.abort())

            domain = socket.gethostbyname(self.root_url.replace("https://", ""))
            try:
                ip = socket.gethostbyname(domain)
                print(f"[DNS OK] {domain}")
            except socket.gaierror:
                print(f"[DNS ERROR] Impossibile risolvere {domain}")

            # Vai alla home e accetta cookie
            await page.goto(self.root_url, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.click("button:has-text(\"Accetta tutti\")", timeout=5000)
            except Exception:
                self.logger.log_info("Nessun banner cookie da accettare.")

            async def safe_goto(url_goto: str, attempt: int = 1) -> bool:
                try:
                    await page.goto(url_goto, wait_until="domcontentloaded", timeout=60000)
                    await page.wait_for_selector("div.nd-mediaObject__content", timeout=15000)
                    return True
                except PlaywrightTimeoutError as ple:
                    self.logger.log_info(f"Timeout su {url} (tentativo {attempt}/{self.max_attempts})")
                    if attempt < self.max_attempts:
                        await asyncio.sleep(5 * attempt)  # back-off
                        return await safe_goto(url, attempt + 1)
                    else:
                        self.logger.log_exception(f"Impossibile caricare {url}", ple)
                        return False
                except Exception as e:
                    self.logger.log_exception(f"Errore generico su {url}", e)
                    return False


            from_page = self.from_page
            for i in range(1, n_pages + 1):
                url = self.base_url + f"{i}"
                success = await safe_goto(url)
                if not success:
                    continue

                await page.mouse.wheel(0, 10000)
                await asyncio.sleep(1)

                html = await page.content()
                file_path = out_dir / f"annuncio_for_training_{from_page}.html"
                from_page += 1
                file_path.write_text(html, encoding="utf-8")
                self.logger.log_info(f"Pagina {i} salvata: {file_path}")

            await browser.close()

        except Exception as e:
            self.logger.log_exception("Errore durante la fase di scraping Playwright", e)


    @alog_exec(logger_name="fetcher", method_name="fetch")
    async def fetch(self, n_pages: int, out_dir: Path):
        self.logger.log_info(f"Avvio fetch: {n_pages} pagine → {out_dir}")
        try:
            async with async_playwright() as pw:
                await self.scrape_data(pw, n_pages, out_dir)
            self.logger.log_info("Download completato.")
        except Exception as e:
            self.logger.log_exception("Errore critico durante fetch()", e)
