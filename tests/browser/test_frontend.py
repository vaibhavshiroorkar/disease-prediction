"""Real browser flows against the API and committed models."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1", reason="set RUN_BROWSER_TESTS=1 and install Chromium",
)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (tmp_path_factory.mktemp("browser") / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "scripts/serve.py"],
            cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=log,
            env={**os.environ, "PORT": str(port), "DATABASE_URL": "", "RATE_LIMIT_PER_MINUTE": "600"},
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail("Browser test server exited; inspect server.log")
                try:
                    with urllib.request.urlopen(f"{url}/ready", timeout=1) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            else:
                pytest.fail("Browser test server did not become ready")
            yield url
        finally:
            process.terminate()
            process.wait(timeout=15)


@pytest.fixture()
def page(site):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(site)
        yield page
        browser.close()
        assert not errors, errors


def test_symptom_example_produces_ranked_results(page):
    from playwright.sync_api import expect

    page.get_by_role("button", name="Use example", exact=True).click()
    page.get_by_role("button", name="Run prediction", exact=True).click()
    result = page.locator("#results").get_by_role("heading", name="Fungal infection", exact=True)
    expect(result).to_be_visible()
    expect(page.locator("#results")).to_contain_text("Model score")
    expect(page.locator("#results")).to_contain_text("not medical advice")


@pytest.mark.parametrize("model", ["Heart", "Diabetes", "Liver", "Dengue"])
def test_clinical_forms_submit_measurements_and_explain_results(page, model):
    from playwright.sync_api import expect

    page.get_by_role("button", name=model, exact=True).click()
    page.get_by_role("button", name="Use example", exact=True).click()
    page.get_by_role("button", name="Run prediction", exact=True).click()
    expect(page.locator("#results")).to_contain_text("Model estimate")
    expect(page.locator("#results")).to_contain_text("Inputs that changed the score")
    expect(page.locator("#results")).to_contain_text("not medical advice")


def test_malaria_upload_reaches_the_image_model(page):
    import io

    from PIL import Image, ImageDraw
    from playwright.sync_api import expect

    image = Image.new("RGB", (120, 120), "black")
    ImageDraw.Draw(image).ellipse((10, 10, 110, 110), fill=(200, 150, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    page.get_by_role("button", name="Malaria", exact=True).click()
    page.get_by_label("Blood cell image").set_input_files(
        {"name": "cell.png", "mimeType": "image/png", "buffer": buffer.getvalue()},
    )
    page.get_by_role("button", name="Run prediction", exact=True).click()
    expect(page.locator("#results")).to_contain_text("Model estimate")
    expect(page.locator("#results")).to_contain_text("Single-cell classification")


def test_request_error_is_readable_and_the_form_recovers(page):
    from playwright.sync_api import expect

    page.route("**/predict/symptoms", lambda route: route.fulfill(
        status=429, content_type="application/json",
        body='{"detail":"Too many requests. Try again shortly."}',
    ))
    page.get_by_role("button", name="Use example", exact=True).click()
    page.get_by_role("button", name="Run prediction", exact=True).click()
    expect(page.get_by_role("alert")).to_contain_text("Too many requests")
    expect(page.get_by_role("button", name="Run prediction", exact=True)).to_be_enabled()


def test_mobile_layout_has_no_horizontal_overflow_and_supports_keyboard(page):
    from playwright.sync_api import expect

    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Symptoms", exact=True).focus()
    page.keyboard.press("Tab")
    page.keyboard.press("Enter")
    expect(page.get_by_label("Chest pain type")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_clearing_symptoms_does_not_reuse_an_old_result(page):
    from playwright.sync_api import expect

    page.get_by_role("button", name="Use example", exact=True).click()
    page.get_by_role("button", name="Run prediction", exact=True).click()
    result = page.locator("#results").get_by_role("heading", name="Fungal infection", exact=True)
    expect(result).to_be_visible()
    page.get_by_role("button", name="Clear inputs", exact=True).click()
    expect(page.locator("#results")).to_contain_text("Your results will appear here")
    page.get_by_role("button", name="Run prediction", exact=True).click()
    expect(page.get_by_role("alert")).to_contain_text("Choose at least one symptom")
