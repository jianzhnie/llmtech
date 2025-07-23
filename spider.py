import logging
import os
import urllib.parse
from datetime import date
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

SITEMAP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{url_entries}
</urlset>"""  # 注意这里去掉了多余的换行

URL_ENTRY_TEMPLATE = """<url>
<loc>{url}</loc>
<lastmod>{lastmod}</lastmod>
<changefreq>{changefreq}</changefreq>
<priority>{priority}</priority>
</url>
"""

# 要扫描的子目录
TARGET_DIRS = [
    'aigc',
    'ascend',
    'diffusion',
    'inference',
    'multimodal',
    'rlhf',
    'ultrascale',
    'rlwiki',
    'toolbox',
]

# 要忽略的文件名
IGNORE_FILES = ['_navbar.md', '_sidebar.md', '_coverpage.md', '_footer.md']


def get_priority(url: str) -> float:
    """根据URL特征确定优先级"""
    if 'README' in url or 'index' in url.lower():
        return 1.0
    elif any(key in url.lower() for key in ['aigc', 'llm', 'rlhf']):
        return 0.9
    return 0.8


def get_changefreq(url: str) -> str:
    """根据URL特征确定更新频率"""
    if 'README' in url or 'index' in url.lower():
        return 'daily'
    return 'weekly'


def generate_urls_from_md_files(docs_dir: str, base_url: str) -> List[str]:
    """生成网站URL列表"""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f'文档目录不存在: {docs_dir}')

    urls = []
    for target_dir in TARGET_DIRS:
        full_path = os.path.join(docs_dir, target_dir)
        if not os.path.exists(full_path):
            logging.warning(f'目录不存在：{full_path}')
            continue

        try:
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
                        logging.debug(f'添加URL: {full_url}')
        except Exception as e:
            logging.error(f'处理目录 {target_dir} 时出错: {str(e)}')

    return sorted(urls)


def main():
    try:
        base_url = 'https://jianzhnie.github.io/llmtech/'
        docs_dir = './docs'
        today = date.today().isoformat()

        # 检查输出目录是否存在
        output_dir = 'docs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        urls = generate_urls_from_md_files(docs_dir, base_url)

        entries = ''.join([
            URL_ENTRY_TEMPLATE.format(url=url,
                                      lastmod=today,
                                      changefreq=get_changefreq(url),
                                      priority=get_priority(url))
            for url in urls
        ])

        sitemap = SITEMAP_TEMPLATE.format(url_entries=entries)

        output_path = os.path.join(output_dir, 'sitemap.xml')
        # 写入文件时不添加任何额外的空行
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(sitemap.strip())  # 使用 strip() 移除多余的空白字符

        logging.info(f'✅ 成功生成 sitemap.xml，包含 {len(urls)} 个URL')

    except Exception as e:
        logging.error(f'生成sitemap时出错: {str(e)}')
        raise


if __name__ == '__main__':
    main()
