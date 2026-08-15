import markdown
import os
import asyncio
from playwright.async_api import async_playwright

def build_html():
    with open('AI_Training_Handbook.md', 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'toc', 'sane_lists']
    )

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Training Handbook</title>
    <!-- KaTeX CSS & JS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

    <style>
        @page {{
            size: A4;
            margin: 16mm 14mm 16mm 14mm;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            line-height: 1.5;
            font-size: 9.8pt;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}

        h1 {{
            font-size: 20pt;
            font-weight: 700;
            color: #0f172a;
            border-bottom: 2.5px solid #2563eb;
            padding-bottom: 6px;
            margin-top: 22px;
            margin-bottom: 14px;
            page-break-after: avoid;
        }}

        h2 {{
            font-size: 14pt;
            font-weight: 600;
            color: #1e3a8a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 18px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}

        h3 {{
            font-size: 11.5pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 10.5pt;
            font-weight: 600;
            color: #334155;
            margin-top: 12px;
            margin-bottom: 4px;
            page-break-after: avoid;
        }}

        p {{
            margin-top: 0;
            margin-bottom: 8px;
        }}

        code {{
            font-family: Consolas, 'Courier New', monospace;
            font-size: 8.5pt;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #e2e8f0;
        }}

        pre {{
            background-color: #0f172a;
            color: #f8fafc;
            padding: 10px 14px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: Consolas, 'Courier New', monospace;
            font-size: 8pt;
            line-height: 1.4;
            margin-top: 6px;
            margin-bottom: 12px;
            page-break-inside: avoid;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
            border: none;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 14px;
            font-size: 8.5pt;
            page-break-inside: avoid;
        }}

        th {{
            background-color: #1e293b;
            color: #ffffff;
            font-weight: 600;
            text-align: left;
            padding: 6px 8px;
            border: 1px solid #334155;
        }}

        td {{
            padding: 5px 8px;
            border: 1px solid #cbd5e1;
        }}

        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        blockquote {{
            border-left: 4px solid #2563eb;
            background-color: #eff6ff;
            margin: 10px 0;
            padding: 8px 14px;
            border-radius: 0 4px 4px 0;
            color: #1e40af;
        }}

        ul, ol {{
            margin-top: 0;
            margin-bottom: 8px;
            padding-left: 20px;
        }}

        li {{
            margin-bottom: 3px;
        }}

        hr {{
            border: 0;
            height: 1px;
            background: #e2e8f0;
            margin: 16px 0;
        }}

        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 14px auto;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            page-break-inside: avoid;
        }}

        .katex-display {{
            margin: 10px 0;
            overflow-x: auto;
            overflow-y: hidden;
        }}
    </style>
</head>
<body>
{html_body}

<script>
    document.addEventListener("DOMContentLoaded", function() {{
        renderMathInElement(document.body, {{
            delimiters: [
                {{left: "$$", right: "$$", display: true}},
                {{left: "$", right: "$", display: false}},
                {{left: "\\\\(", right: "\\\\)", display: false}},
                {{left: "\\\\[", right: "\\\\]", display: true}}
            ],
            throwOnError: false
        }});
    }});
</script>
</body>
</html>"""

    with open('AI_Training_Handbook.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    print("Generated AI_Training_Handbook.html successfully.")

async def render_pdf():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        cwd = os.path.abspath('.')
        html_url = f"file:///{cwd.replace(os.sep, '/')}/AI_Training_Handbook.html"
        print(f"Loading {html_url} in headless Chromium...")
        await page.goto(html_url, wait_until="networkidle")
        # Give KaTeX a moment to finish rendering math elements
        await page.wait_for_timeout(2000)
        
        pdf_path = os.path.join(cwd, "AI_Training_Handbook.pdf")
        print(f"Generating PDF to {pdf_path} ...")
        await page.pdf(
            path=pdf_path,
            format="A4",
            margin={
                "top": "16mm",
                "bottom": "16mm",
                "left": "14mm",
                "right": "14mm"
            },
            print_background=True
        )
        await browser.close()
        print(f"PDF generated successfully: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

if __name__ == '__main__':
    build_html()
    asyncio.run(render_pdf())
