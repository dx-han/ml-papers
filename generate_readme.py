import os

TITLE = 'ML papers'
IGNORE = ['.git', 'LICENSE', 'README.md', 'generate_readme.py', 'README0.md']


def find_file_folder(path):
    f_list = [folder for folder in os.listdir(path) if folder not in IGNORE]
    return f_list


with open('README.md', 'w+', encoding='utf8') as f:
    f.write('# ' + TITLE + '\n')
    f.write('\n')
    folder_list = find_file_folder(os.getcwd())
    for folder in folder_list:
        f.write('### ' + folder + '\n')
        file_list = find_file_folder(os.path.join(os.getcwd(), folder))
        for idx, file in enumerate(file_list):
            print(idx)
            f.write('{}. {} \n'.format(idx + 1, file))
