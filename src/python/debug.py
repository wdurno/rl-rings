from time import sleep
import argparse 

parser = argparse.ArgumentParser(description='run a sleeping process') 
parser.add_argument('--silent', dest='silent', default=False, action='store_true', help='do no print') 
args = parser.parse_args() 

def interactive_debugging_mode():
    if not args.silent: 
        print('starting in interactive debugging mode...')
    while True:
        if not args.silent: 
            print('sleeping 60 seconds...')
        sleep(60)
        pass
    pass

if __name__ == '__main__':
    interactive_debugging_mode()
    pass 
