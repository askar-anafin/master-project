import markdown
import os

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # CSS for styling (IEEE-like)
    css = """
    <style>
        body { font-family: 'Times New Roman', Times, serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; color: #333; }
        h1 { font-size: 24px; text-align: center; margin-bottom: 20px; }
        h2 { font-size: 18px; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }
        h3 { font-size: 16px; font-weight: bold; margin-top: 20px; }
        h4 { font-size: 14px; font-style: italic; margin-top: 15px; }
        p { margin-bottom: 10px; text-align: justify; }
        ul { margin-bottom: 10px; }
        li { margin-bottom: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
        .print-btn { display: block; margin: 20px auto; padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; border-radius: 5px; }
        @media print {
            .print-btn { display: none; }
            body { max-width: 100%; margin: 0; padding: 0; }
        }
    </style>
    """
    
    html_content = markdown.markdown(text, extensions=['tables'])
    
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Project Report</title>
        {css}
    </head>
    <body>
        <button class="print-btn" onclick="window.print()">Print to PDF 🖨️</button>
        {html_content}
    </body>
    </html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"Converted {md_path} to {html_path}")

if __name__ == "__main__":
    convert_md_to_html('REPORT.MD', 'REPORT.html')
