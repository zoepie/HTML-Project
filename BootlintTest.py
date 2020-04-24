import os
import subprocess
from subprocess import check_output, Popen, PIPE

file_path = []
def get_file_path():
    for i in range(100):
        file = 'Z:/project/html_pro/dataset/%s/%d.txt' % ('test', i)
        file_path.append(file)
    print(file_path)


def run_command():
    get_file_path()
    counter = 0
    for i in file_path:
        counter += 1
        p = Popen(['bootlint', i], stdout=PIPE, stderr=PIPE, shell=True)
        output, error = p.communicate()
        if p.returncode != 0:
            print(output.decode("utf-8"))
        with open('dataset/testresults/' + str(counter) + '.txt', 'w') as f:
            f.write(output.decode("utf-8"))


if __name__ == '__main__':

    run_command()
    print()