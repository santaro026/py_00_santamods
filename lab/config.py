from pathlib import Path
import os
import re

def find_projectroot(curdir=None, markfile=".myprojectroot"):
    if curdir == None: curdir = Path(os.getcwd())
    while True:
        if (curdir / markfile).exists(): break
        if curdir == curdir.parent:
            print(f"**** markfile was not found.")
            curdir = Path(__file__).parent
            break
        curdir = curdir.parent
    return curdir

def get_versioninfo(scrdir):
    releasenote = str(list(scrdir.glob('released_at_*'))[0])
    releasedate = re.search(r'released_at_(\d{6})', releasenote).group(1)
    version = re.search(r'(v_\d+_\d+_\d+)', releasenote).group(1)
    return releasedate, version

scrdir = Path(__file__).resolve().parent
ROOT = find_projectroot(scrdir)
RELEASEDATE, VERSION = get_versioninfo(scrdir)

if __name__ == '__main__':
    print('---- test ----')
    print(f'ROOT: {ROOT}')
    print(f'RELEASEDATE, {RELEASEDATE}')
    print(f'VERSION, {VERSION}')


