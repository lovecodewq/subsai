DOCKER_IMAGE=subsai:20240411
docker rm subsai
docker run -it --name subsai \
-v /home/wenqiangli/code/subsai:/subsai \
-w /subsai -v /home/wenqiangli/code/main/expand_data:/data \
"${DOCKER_IMAGE}" /bin/bash