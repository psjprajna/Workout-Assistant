"""Download the Jefit data export with a headless browser.

Jefit has no export API: the export is a button on a page behind the login
(EXPORT_URL), so this signs in as the user and clicks it. The form is found by its
labels and the button by its text, not by CSS classes, so a restyle doesn't break it;
a reworded or reorganised page will, and then the error says which step failed.
Playwright is imported inside the function so the API starts without it.
"""

import re
from pathlib import Path

LOGIN_URL = "https://www.jefit.com/login"
EXPORT_URL = "https://www.jefit.com/my-jefit/settings/exportData.php"
EXPORT_START = "### SETTING"  # first line of every export (importing/sections.py)


class JefitFetchError(Exception):
    """The export could not be downloaded; the message says which step failed."""


async def download_export(
    email: str, password: str, dest_dir: Path, headless: bool = True, timeout_s: float = 90
) -> Path:
    """Sign in to Jefit, download the export into dest_dir and return its path."""
    from playwright.async_api import Error as PlaywrightError
    from playwright.async_api import async_playwright

    timeout = timeout_s * 1000
    dest_dir.mkdir(parents=True, exist_ok=True)
    step = "starting the browser"
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=headless)
        except PlaywrightError as error:
            raise JefitFetchError(
                f"{step}: {error} (run `uv run playwright install chromium` once)"
            ) from error
        try:
            page = await browser.new_page()
            page.set_default_timeout(timeout)
            step = "opening the login page"
            await page.goto(LOGIN_URL)
            cookies = page.get_by_role("button", name="Accept")
            if await cookies.count():  # the cookie banner can sit over the form
                await cookies.first.click()
            step = "filling in the login form"
            await page.get_by_label(re.compile(r"e-?mail|user ?name", re.I)).first.fill(email)
            await page.locator("input[type=password]").first.fill(password)
            step = "signing in (wrong email or password?)"
            # Exactly "Log In": the page also has "Sign in with Apple".
            await page.get_by_role("button", name=re.compile(r"^\s*log ?in\s*$", re.I)).click()
            await page.wait_for_url(lambda url: "/login" not in url)
            step = "opening the export page"
            await page.goto(EXPORT_URL)
            step = "clicking the export button"
            wanted = re.compile(r"export|download", re.I)
            control = page.get_by_role("button", name=wanted).or_(
                page.get_by_role("link", name=wanted)
            )
            async with page.expect_download() as pending:
                await control.first.click()
            download = await pending.value
            path = dest_dir / download.suggested_filename
            await download.save_as(path)
        except PlaywrightError as error:
            raise JefitFetchError(f"{step}: {error}") from error
        finally:
            await browser.close()
    if not path.read_text(encoding="utf-8", errors="replace").startswith(EXPORT_START):
        raise JefitFetchError(f"the downloaded file {path.name} is not a Jefit export")
    return path
