DOCKERCOMPOSE_VERSION=1.8.0
KERNEL_NAME=$(uname -s)
KERNEL_ARQ=$(uname -m)
curl -L https://github.com/docker/compose/releases/download/"$DOCKERCOMPOSE_VERSION"/docker-compose-"$KERNEL_NAME"-"$KERNEL_ARQ" > docker-compose
mv docker-compose /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
