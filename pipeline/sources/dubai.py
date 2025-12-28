from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.async_api import Browser, Page, async_playwright

logger = logging.getLogger(__name__)

DUBAI_URL = "https://esupply.dubai.gov.ae/"
DUBAI_EXCEL_NAME = "dubai_tenders.xls"


class DubaiScraper:
    def __init__(self, excel_path: Optional[Path] = None, headless: bool = True):
        self.excel_path = Path(excel_path) if excel_path else Path(DUBAI_EXCEL_NAME)
        self.headless = headless

    async def _scrape_async(self) -> pd.DataFrame:
        logger.info("[DXB] Starting Dubai scraping...")
        async with async_playwright() as playwright:
            browser: Browser = await playwright.chromium.launch(headless=self.headless)
            context = await browser.new_context(accept_downloads=True)
            page: Page = await context.new_page()
            await page.goto(DUBAI_URL)

            if await page.is_visible("text=Your session is invalid or expired"):
                await page.click("text=Main Page")
                await page.wait_for_load_state("networkidle")

            await page.evaluate("window.scrollTo(0, 2000)")
            await page.click("text=Search Now")

            await page.wait_for_selector("text=Published Opportunities", timeout=60000)
            await page.wait_for_selector("table", timeout=60000)

            try:
                await page.get_by_label("Other Actions Menu").click()
            except Exception:
                await page.locator("button[aria-label='Other Actions Menu']").click()

            async with page.expect_download() as download_info:
                await page.click("text=Export List")
            download = await download_info.value

            temp_path = await download.path()
            if not temp_path:
                raise RuntimeError("Dubai export did not provide a download path")

            save_path = os.path.abspath(self.excel_path)
            os.replace(temp_path, save_path)
            await browser.close()

        raw_sheet = pd.read_excel(save_path, header=None)
        header_row = raw_sheet.apply(lambda row: row.astype(str).str.contains("Project Code", case=False)).any(axis=1).idxmax()
        df_dubai = pd.read_excel(save_path, header=header_row)

        df_dubai.dropna(axis=1, how="all", inplace=True)
        df_dubai = df_dubai[df_dubai[df_dubai.columns[0]] != "Project Code"]
        df_dubai = df_dubai[~df_dubai[df_dubai.columns[0]].astype(str).str.contains("Site Name|Locale|Export Time", case=False, na=False)]
        df_dubai = df_dubai.applymap(lambda value: value.strip() if isinstance(value, str) else value)

        df_dubai["Source"] = "Dubai"
        if "Project Code" in df_dubai.columns:
            df_dubai["reference_no"] = df_dubai["Project Code"].astype(str).str.strip()
        if "Project Title" in df_dubai.columns:
            df_dubai["tender_title"] = df_dubai["Project Title"].astype(str).str.strip()

        logger.info("[DXB] Scraped Dubai rows: %s", len(df_dubai))
        return df_dubai

    def scrape(self) -> pd.DataFrame:
        return asyncio.run(self._scrape_async())
