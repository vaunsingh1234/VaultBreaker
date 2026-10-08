"""Automated browser capture script for VaultBreaker CyFocus UI."""

import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright

async def capture():
    out_dir = Path("docs/RESULTS/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        print("Navigating to http://localhost:8501/...")
        await page.goto("http://localhost:8501/", wait_until="networkidle")
        await page.wait_for_timeout(4000)

        # 1. Capture Dashboard
        print("Capturing Dashboard...")
        await page.screenshot(path=str(out_dir / "cyfocus_dashboard.png"), full_page=True)
        print("  -> Saved cyfocus_dashboard.png")

        # 2. Navigate to Scans (Findings Kanban)
        print("Navigating to Scans...")
        # Find button with text 'Scans'
        scans_btn = page.locator("button:has-text('Scans')").first
        if await scans_btn.count() > 0:
            await scans_btn.click()
            await page.wait_for_timeout(3000)
            await page.screenshot(path=str(out_dir / "cyfocus_scans_kanban.png"), full_page=True)
            print("  -> Saved cyfocus_scans_kanban.png")

        # 3. Navigate to Scan Details (Incident Details)
        print("Navigating to Scan Details...")
        details_btn = page.locator("button:has-text('Scan Details')").first
        if await details_btn.count() > 0:
            await details_btn.click()
            await page.wait_for_timeout(3000)
            await page.screenshot(path=str(out_dir / "cyfocus_scan_details.png"), full_page=True)
            print("  -> Saved cyfocus_scan_details.png")

        # 4. Navigate to New Scan
        print("Navigating to New Scan...")
        new_scan_btn = page.locator("button:has-text('New scan +')").first
        if await new_scan_btn.count() > 0:
            await new_scan_btn.click()
            await page.wait_for_timeout(2000)
            await page.screenshot(path=str(out_dir / "cyfocus_new_scan.png"), full_page=True)
            print("  -> Saved cyfocus_new_scan.png")

        # 5. Navigate to Models
        print("Navigating to Models...")
        models_btn = page.locator("button:has-text('Models')").first
        if await models_btn.count() > 0:
            await models_btn.click()
            await page.wait_for_timeout(2000)
            await page.screenshot(path=str(out_dir / "cyfocus_models.png"), full_page=True)
            print("  -> Saved cyfocus_models.png")

        await browser.close()
        print("All CyFocus screenshots successfully captured!")

if __name__ == "__main__":
    asyncio.run(capture())
