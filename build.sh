IMAGE_NAME=tybalex/opni-inference:dev-gpu
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
