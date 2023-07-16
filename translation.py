import os
import re
import shutil
import nbformat
from nbconvert import MarkdownExporter

# 设置要搜索的目录和要替换的文件夹
search_dir = '.'
replace_dir = 'trans'

# 定义用于查找和替换字符串的正则表达式
pattern = re.compile(r'\(([^()]*?)\.ipynb\)')

# 获取当前目录中的所有 ipynb 文件
for file in os.listdir(search_dir):
    if file.endswith('.ipynb'):
        # 读取 ipynb 文件内容
        with open(file, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # 创建一个 Markdown 导出器并导出内容
        md_exporter = MarkdownExporter()
        md_content, _ = md_exporter.from_notebook_node(nb)
        
        # 将导出的内容写入到一个新的 markdown 文件中
        md_file = file.replace('.ipynb', '.md')
        md_path = os.path.join(replace_dir, md_file)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

# 遍历搜索目录及其子目录
for root, dirs, files in os.walk(search_dir):
    for file in files:
        # 检查文件是否为 ipynb 文件
        if file.endswith('.ipynb'):
            # 获取 ipynb 文件的完整路径
            ipynb_path = os.path.join(root, file)
            # 获取与之同名的 md 文件名
            md_file = file.replace('.ipynb', '.md')
            # 获取 md 文件的完整路径
            md_path = os.path.join(replace_dir, md_file)
            # 检查 md 文件是否存在
            if os.path.exists(md_path):
                # 删除原始的 ipynb 文件
                os.remove(ipynb_path)
                # 将 md 文件移动到 ipynb 文件所在的位置
                shutil.move(md_path, os.path.join(root, md_file))
                print(f'Replaced {ipynb_path} with {md_path}')
        # 检查文件是否为 md 文件
        elif file.endswith('.md'):
            # 获取 md 文件的完整路径
            md_path = os.path.join(root, file)
            # 读取 md 文件的内容
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 使用正则表达式查找和替换字符串
            new_content = pattern.sub(r'(\1.md)', content)
            # 如果内容发生更改，则写回文件
            if new_content != content:
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f'Updated {md_path}')
