docker build --build-arg="GID=${UID}" --build-arg="UID=${UID}" --build-arg="NAME=${USER}" -t ${USER}_$(basename $(dirname "$PWD")) ../docker_scripts