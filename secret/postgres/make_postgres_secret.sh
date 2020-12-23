python -c 'import os,base64; print(base64.urlsafe_b64encode(os.urandom(16)).decode())' > ${repo_dir}/secret/postgres/postgres-secret 
