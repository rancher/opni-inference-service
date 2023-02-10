IMAGE_NAME=tybalex/opni-inference:devb
docker build . -t $IMAGE_NAME -f ./Dockerfile

docker push $IMAGE_NAME
