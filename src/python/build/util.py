import os 
from subprocess import Popen, PIPE 

## constants 
## text color constants
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
NC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def run(cmd: str, stdin: str=None, os_system: bool=False, return_stdout=True):
    'Execute a string as a blocking, exception-raising system call'
    ## verify assumptions 
    if type(cmd) != str:
        raise ValueError('`cmd` must be a string!')
    ## execute 
    print(OKCYAN+cmd+NC)
    if stdin is None: 
        ## no stdin 
        if not os_system:
            proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            exit_code = proc.wait() 
            stdout = proc.stdout.read().decode() 
            stderr = proc.stderr.read().decode() 
        else:
            exit_code = os.system(cmd)
            stdout = 'not captured'
            stderr = 'not captured'
    else:
        ## apply stdin 
        if type(stdin) not in [str, bytes]:
            raise ValueError('STDIN must be str or bytes!')
        if type(stdin) == str:
            ## convert to bytes
            stdin = stdin.encode() 
        proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, stdin=PIPE) 
        stdout, stderr = proc.communicate(stdin)
        stdout = stdout.decode() 
        stderr = stderr.decode() 
        exit_code = proc.returncode 
    if exit_code != 0:
        print(OKCYAN+'STDOUT: '+stdout+NC) 
        print(OKCYAN+'STDERR: '+stderr+NC) 
        raise OSError(exit_code)
    if return_stdout:
        return stdout 
    pass

