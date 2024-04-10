docker rm subsai
docker run -it --name subsai \
-v /home/wenqiangli/code/subsai:/subsai \
-v /Users/wenqiangli/code/subsai:/subsai \
-w /subsai -v /home/wenqiangli/code/main/expand_data:/data \
-w /subsai -v
subsai_base:20240408 /bin/bash