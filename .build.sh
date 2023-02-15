IMAGE_NAME=tybalex/opni-inference:deve2
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
