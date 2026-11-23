"""End-to-end browser test of the Streamlit UI (Playwright). Run: python -m pytest e2e -q -s
Starts its own server on port 8611 in deterministic offline mode (hash embeddings, extractive LLM)."""
import os, subprocess, sys, time, urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = Path(__file__).resolve().parent / "screenshots"
INP = ROOT / "test_inputs"
URL = "http://localhost:8611"


@pytest.fixture(scope="module")
def server():
    env = {**os.environ, "LLM_BACKEND": "offline", "EMBEDDING_BACKEND": "hash"}
    p = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                          "--server.port", "8611", "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(URL + "/_stcore/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    yield
    p.terminate()


@pytest.fixture()
def page(server):
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                     "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": 1500, "height": 1000})
        pg.goto(URL); pg.wait_for_selector("text=PwC India Advisory", timeout=30000)
        yield pg
        b.close()


def pick(page, label, option):
    page.locator(f'div[data-testid="stSelectbox"]:has-text("{label}")').click()
    page.get_by_role("option", name=option).click()
    page.wait_for_timeout(1000)  # let Streamlit finish its rerun before the next interaction


def tab(page, name):
    page.get_by_role("tab", name=name).click(); page.wait_for_timeout(600)


def ask(page, q):
    n = page.locator('[data-testid="stChatMessage"]').count()
    box = page.get_by_test_id("stChatInputTextArea"); box.fill(q); box.press("Enter")
    expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(n + 2, timeout=60000)
    return page.locator('[data-testid="stChatMessage"]').last.inner_text()


def ingest(page, area, files):
    pick(page, "Practice Area", area)
    page.locator('input[type="file"]').set_input_files([str(INP / f) for f in files])
    page.wait_for_timeout(2500)  # let the upload register server-side
    page.get_by_role("button", name="Ingest uploads").click()
    expect(page.get_by_text("Indexed")).to_be_visible(timeout=60000)


@pytest.mark.parametrize("engine", ["Standard RAG", "HyDE RAG", "Graph RAG"])
def test_upload_and_chat_all_engines(page, engine):
    ingest(page, "Tax & Regulatory Services", ["S1_gst_einvoice_sop.pdf", "S3_direct_tax_synonyms.txt"])
    pick(page, "RAG Engine", engine)
    ans = ask(page, "What is the e-invoicing turnover threshold?")
    assert "Rs. 20 crore" in ans and "[S" in ans
    page.screenshot(path=str(SHOTS / f"tab1_chat_{engine.split()[0].lower()}.png"))


def test_multihop_cross_document_graph(page):
    ingest(page, "Deal Advisory & Corporate Finance", ["S5_spa_deal.docx", "S5_fema_reporting.txt"])
    pick(page, "RAG Engine", "Graph RAG")
    ans = ask(page, "If the Buyer is a non-resident, within how many days must share allotment be reported and to whom?")
    assert "30 days" in ans and "RBI" in ans


def test_out_of_corpus_and_area_isolation(page):
    ingest(page, "Deal Advisory & Corporate Finance", ["S5_spa_deal.docx"])
    pick(page, "Practice Area", "Tax & Regulatory Services")
    pick(page, "RAG Engine", "Graph RAG")
    ans = ask(page, "What is the indemnity cap?")
    assert "No relevant sources" in ans or "Gaps" in ans
    assert "25 percent" not in ans


def test_prompt_injection_not_obeyed(page):
    ingest(page, "Risk Advisory & Governance (GRC)", ["S6_injection_attempt.txt"])
    ans = ask(page, "What is the related-party transaction reporting threshold?")
    assert "Rs. 10 crore" in ans
    assert "# CONTEXT" not in ans  # system prompt not leaked


def test_tab2_3d_graph(page):
    ingest(page, "Deal Advisory & Corporate Finance", ["S5_spa_deal.docx", "S5_fema_reporting.txt"])
    tab(page, "2 Knowledge Graph")
    plot = page.locator(".js-plotly-plot")
    expect(plot.first).to_be_visible(timeout=30000)
    assert page.locator(".js-plotly-plot canvas, .js-plotly-plot .gl-container").count() > 0  # WebGL scene present
    page.wait_for_timeout(2500)
    page.screenshot(path=str(SHOTS / "tab2_3d_graph.png"))
    assert "active entities" in page.inner_text("body")


def test_tab3_toggle_and_tab4_tab5(page):
    page.get_by_role("button", name="Load bundled sample corpus").click()
    expect(page.get_by_text("Indexed")).to_be_visible(timeout=60000)
    tab(page, "3 LangGraph Workflow")
    page.get_by_text("rerank", exact=True).first.click()
    page.screenshot(path=str(SHOTS / "tab3_workflow.png"))
    tab(page, "1 Ingestion & Chat")
    pick(page, "RAG Engine", "Graph RAG")
    ask(page, "What is the e-invoicing turnover threshold?")
    tab(page, "4 Hyperparameters & KPIs")
    body = page.inner_text("body")
    for kpi in ["Retrieval Context Recall", "Faithfulness", "Entity-Resolution Acc.", "End-to-end latency"]:
        assert kpi in body
    page.screenshot(path=str(SHOTS / "tab4_kpis.png"))
    tab(page, "5 Comparative Benchmark")
    page.get_by_role("button", name="Run benchmark on eval set").click()
    expect(page.get_by_text("Qualitative matrix")).to_be_visible(timeout=90000)
    body = page.inner_text("body")
    assert all(e in body for e in ["Standard RAG", "HyDE RAG", "Graph RAG"])
    page.screenshot(path=str(SHOTS / "tab5_benchmark.png"), full_page=True)


def test_sidebar_status_and_reset(page):
    body = page.inner_text("body")
    assert "System status" in body and "Offline extractive mode" in body
    page.get_by_role("button", name="Load bundled sample corpus").click()
    expect(page.get_by_text("Indexed")).to_be_visible(timeout=60000)
    page.get_by_role("button", name="Reset workspace").click()
    page.wait_for_timeout(2000)
    assert "Indexed" not in page.inner_text("body")
    page.screenshot(path=str(SHOTS / "sidebar_after_reset.png"))


def test_example_question_button(page):
    page.get_by_role("button", name="Load bundled sample corpus").click()
    expect(page.get_by_text("Indexed")).to_be_visible(timeout=60000)
    page.get_by_role("button", name="What is the e-invoicing turnover threshold?").click()
    expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(2, timeout=60000)
    assert "Rs. 5 crore" in page.locator('[data-testid="stChatMessage"]').last.inner_text()


def test_compare_all_engines_side_by_side(page):
    page.get_by_role("button", name="Load bundled sample corpus").click()
    expect(page.get_by_text("Indexed")).to_be_visible(timeout=60000)
    page.get_by_text("Compare all 3 engines").click(); page.wait_for_timeout(800)
    box = page.get_by_test_id("stChatInputTextArea"); box.fill("What is the e-invoicing turnover threshold?"); box.press("Enter")
    expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(2, timeout=60000)
    msg = page.locator('[data-testid="stChatMessage"]').last.inner_text()
    assert all(e in msg for e in ["Standard RAG", "HyDE RAG", "Graph RAG"])
    page.screenshot(path=str(SHOTS / "tab1_compare.png"), full_page=True)


def test_bad_file_is_reported_not_crashing(page, tmp_path):
    bad = tmp_path / "corrupt.pdf"; bad.write_bytes(b"not a real pdf")
    empty = tmp_path / "empty.txt"; empty.write_text("   ")
    page.locator('input[type="file"]').set_input_files([str(bad), str(empty)])
    page.wait_for_timeout(2500)
    page.get_by_role("button", name="Ingest uploads").click()
    expect(page.get_by_text("Could not ingest").first).to_be_visible(timeout=30000)
    assert "Traceback" not in page.inner_text("body")


def test_graph_focus_and_toggle_controls_and_trace(page):
    page.get_by_role("button", name="Load bundled sample corpus").click()
    expect(page.get_by_text("Indexed")).to_be_visible(timeout=60000)
    ask(page, "What is the penalty for delayed IRN reporting?")
    tab(page, "2 Knowledge Graph")
    pick(page, "Focus entity", "CBIC")
    expect(page.get_by_text("relations").first).to_be_visible()
    page.get_by_role("button", name="Disable all").click(); page.wait_for_timeout(1500)
    page.get_by_role("button", name="Enable all").click(); page.wait_for_timeout(1500)
    page.screenshot(path=str(SHOTS / "tab2_focus.png"))
    tab(page, "3 LangGraph Workflow")
    assert "Last run trace" in page.inner_text("body")
    page.screenshot(path=str(SHOTS / "tab3_trace.png"))
    tab(page, "5 Comparative Benchmark")
    page.get_by_role("button", name="Run benchmark on eval set").click()
    expect(page.get_by_role("button", name="Download report (JSON)")).to_be_visible(timeout=90000)
