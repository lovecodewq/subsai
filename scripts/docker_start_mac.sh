docker rm subsai
docker run -itd --name subsai \
-v /Users/wenqiangli/code/subsai:/subsai \
-w /subsai \
subsai:20240410 /bin/bash