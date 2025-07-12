import os
import urllib.parse
from datetime import date

SITEMAP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{url_entries}
</urlset>
"""

URL_ENTRY_TEMPLATE = """<url>
<loc>{url}</loc>
<lastmod>{lastmod}</lastmod>
<changefreq>weekly</changefreq>
<priority>{priority}</priority>
</url>
"""

# 要扫描的子目录
TARGET_DIRS = ['aigc', 'ascend', 'rlwiki', 'toolbox']

# 要忽略的文件名
IGNORE_FILES = ['_navbar.md', '_sidebar.md', '_coverpage.md', '_footer.md']


def generate_urls_from_md_files(docs_dir, base_url):
    urls = []
    for target_dir in TARGET_DIRS:
        full_path = os.path.join(docs_dir, target_dir)
        if not os.path.exists(full_path):
            print(f'⚠️ 路径不存在：{full_path}')
            continue

        for root, dirs, files in os.walk(full_path):
            # 忽略以 _ 开头的目录
            dirs[:] = [d for d in dirs if not d.startswith('_')]

            for file in files:
                if file.endswith('.md') and file not in IGNORE_FILES:
                    rel_path = os.path.relpath(os.path.join(root, file),
                                               docs_dir)
                    rel_path = rel_path.replace('.md', '')
                    rel_path = rel_path.replace('\\', '/')
                    encoded_path = urllib.parse.quote(rel_path)
                    full_url = f'{base_url}#{encoded_path}'
                    urls.append(full_url)

    return sorted(urls)


def main():
    base_url = 'https://jianzhnie.github.io/llmtech/'
    docs_dir = './docs'
    today = date.today().isoformat()

    urls = generate_urls_from_md_files(docs_dir, base_url)

    entries = ''.join([
        URL_ENTRY_TEMPLATE.format(
            url=url,
            lastmod=today,
            priority=1.0 if 'index' in url.lower() else 0.8) for url in urls
    ])

    sitemap = SITEMAP_TEMPLATE.format(url_entries=entries)

    with open('docs/sitemap.xml', 'w', encoding='utf-8') as f:
        f.write(sitemap)

    print(f'✅ 已生成 {len(urls)} 条 URL 到 sitemap.xml')


if __name__ == '__main__':
    main()
