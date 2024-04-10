docker rm subsai
docker run -it --name subsai \
-v /Users/wenqiangli/code/subsai:/subsai \
-w /subsai \
subsai_base:20240408 /bin/bash