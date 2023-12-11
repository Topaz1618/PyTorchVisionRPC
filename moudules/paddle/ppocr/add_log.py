import logging
logging.basicConfig(filename='file_calls.log', level=logging.INFO)
logging.info(f'File {__file__} was called.')
import os

code_to_insert = '''import logging
logging.basicConfig(filename='file_calls.log', level=logging.INFO)
logging.info(f'File {__file__} was called.')
'''

dir_path = '.'


for root, dirs, files in os.walk(dir_path):
    for file in files:
        # 只处理 .py 文件
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(code_to_insert.rstrip('\r\n') + '\n' + content)
