docker rm subsai
docker run -it --name subsai \
-v /home/wenqiangli/code/subsai:/subsai \
-w /subsai -v /home/wenqiangli/code/main/expand_data:/data \
subsai_base_20240409 /bin/bash