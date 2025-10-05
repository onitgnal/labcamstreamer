from playwright.sync_api import sync_playwright, Page, expect
import re

def run_verification(page: Page):
    """
    This script verifies that the cuts plots are rendered correctly after the coordinate system fix.
    """
    # 1. Navigate to the application
    page.goto("http://localhost:5001")

    # 2. Enable the camera
    camera_toggle = page.locator("#cameraToggle")
    expect(camera_toggle).to_be_visible()
    camera_toggle.check()

    # Wait for the stream to load
    stream_img = page.locator("#stream")
    expect(stream_img).to_have_attribute("src", re.compile(r"/video_feed\?ts=\d+"), timeout=10000)


    # 3. Create an ROI by dragging on the live view
    overlay = page.locator("#roiOverlay")
    overlay.drag_to(overlay, source_position={"x": 100, "y": 100}, target_position={"x": 300, "y": 300})

    # 4. Wait for the per-ROI panel to appear and take a screenshot
    per_roi_panel = page.locator("#perRoiPanels")
    expect(per_roi_panel).to_be_visible(timeout=10000)

    # Take a screenshot of the panel
    per_roi_panel.screenshot(path="jules-scratch/verification/verification.png")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            run_verification(page)
        finally:
            browser.close()

if __name__ == "__main__":
    main()