IMAGE_NAME=tybalex/opni-inference:devc7
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
