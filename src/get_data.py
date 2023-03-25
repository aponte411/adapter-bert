import wget
import os
import subprocess

def main():
    # The URL for the dataset zip file.
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    # Download the file (if we haven't already)
    if not os.path.exists('./cola_public_1.1.zip'):
        wget.download(url, './cola_public_1.1.zip')
    if not os.path.exists('./cola_public/'):
        subprocess.run("unzip cola_public_1.1.zip",shell=True)
    print("Dataset downloaded!")

if __name__=="__main__":
    main()
