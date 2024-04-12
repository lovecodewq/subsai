#!/usr/bin/env bash

if [ -f /.dockerenv ]; then
    echo "Please run this script on the host system."
    exit 1
fi
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(realpath "$BASE_DIR/../")"
DOCKER_FILE="${ROOT_DIR}/docker/subsai_base/Dockerfile"
if [ ! -f $DOCKER_FILE ];then
    echo "Docker file ${DOCKER_FILE} is not exist !";
    exit 1
fi

function build() {
  TEMP_DIR="/tmp/subs_ai_base"
  mkdir -p "$TEMP_DIR"
  cp "$DOCKER_FILE" "$TEMP_DIR"
  cp "${ROOT_DIR}/docker/subsai_base/requirements.txt" "$TEMP_DIR"
  cd "$TEMP_DIR" || exit 1
  if ! docker build  -t subsai_base .  ; then
    error "Failed to build subsai base docker!"
    exit 1
  fi
  rm -r "$TEMP_DIR"
}

build
cd "${ROOT_DIR}" || exit 1
