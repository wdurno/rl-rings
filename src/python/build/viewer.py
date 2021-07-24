from build.util import run

def viewer_deploy(root, conf, interactive_mode=True): 
    ## get image name 
    cmd1 = f'cat {root}/secret/acr/server'
    acr_server = run(cmd1, return_stdout=True)
    image_name = acr_server + '/' + conf['image_name'] 
    ## apply helm chart with values 
    cmd2 = f'helm upgrade viewer {root}/src/helm/viewer --install '+\
            f'-f {root}/src/helm/viewer/values.yaml '+\
            f'--set "image={image_name}" '+\
            f'--set "interactive_mode={interactive_mode}"'
    run(cmd2) 
    pass 

